# PP-OCRv6 OCR Service

基于 **PP-OCRv6 small** 的轻量 OCR HTTP 服务：单文件 FastAPI 实现、纯 CPU 推理，适合验证码与短文本识别。

## 特性

- **小**：识别模型仅 21 MB（`PP-OCRv6_small_rec` + 18708 字符词典），常驻内存约 150 MB
- **快**：单张 320×48 图片约 **10 ms**（Intel i5-9500，batch=8 时 126 张/秒）
- **纯 CPU**：无需 GPU / CUDA / OpenVINO，`onnxruntime` 即可
- **接口简单**：两个端点，兼容常见 PT 工具链的 `/captcha/*` 约定
- **可替换**：与 media-saber 的 paddle-ocr 容器接口一致，可直接替换

## 接口

### `GET /captcha/health`

```json
{
  "status": "ok",
  "ocr_service_initialized": true,
  "model": "rec.onnx",
  "timestamp": "2026-11-07T20:35:10Z"
}
```

### `POST /captcha/base64`

请求：

```json
{
  "base64_img": "<裸 base64，不带 data: 前缀>",
  "preprocess": true
}
```

响应：

```json
{
  "result": "6G2H5A",
  "confidence": 0.9327,
  "processing_time": 0.008
}
```

- `result`：识别文本（已做全角 → 半角归一化）
- `confidence`：CTC 解码的平均置信度
- `processing_time`：服务端处理耗时（秒）

## 快速开始

### 1. 下载模型

HuggingFace 直连不稳定时用镜像站：

```bash
# 模型（21 MB）
curl -fL -o models/rec.onnx \
  "https://hf-mirror.com/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/main/inference.onnx"

# 词典（从 inference.yml 提取 character_dict）
curl -fL -o /tmp/inference.yml \
  "https://hf-mirror.com/PaddlePaddle/PP-OCRv6_small_rec_onnx/resolve/main/inference.yml"
python3 - <<'PY'
import yaml
d = yaml.safe_load(open('/tmp/inference.yml', encoding='utf-8'))
chars = d['PostProcess']['character_dict']
open('models/dict.txt', 'w', encoding='utf-8').write('\n'.join(chars) + '\n')
print('字符数:', len(chars))
PY
```

### 2. 构建镜像

```bash
docker build -t ocr-v6-small:latest .
```

### 3. 运行

```bash
docker run -d --name ocr-v6 \
  -p 9899:9899 \
  -v $(pwd)/models:/app/models \
  --restart unless-stopped \
  ocr-v6-small:latest
```

或直接用 docker compose：

```yaml
services:
  ocr-v6:
    build: .
    container_name: ocr-v6
    ports:
      - "9899:9899"
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

### 4. 验证

```bash
curl -s http://127.0.0.1:9899/captcha/health

python3 - <<'PY'
import base64, json, urllib.request
raw = open('captcha.png', 'rb').read()
body = json.dumps({
    "base64_img": base64.b64encode(raw).decode(),
    "preprocess": True,
}).encode()
req = urllib.request.Request(
    'http://127.0.0.1:9899/captcha/base64',
    data=body, headers={'Content-Type': 'application/json'})
print(json.loads(urllib.request.urlopen(req, timeout=20).read()))
PY
```

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL_PATH` | `/app/models/rec.onnx` | ONNX 模型路径 |
| `DICT_PATH` | `/app/models/dict.txt` | 词典路径（每行一个字符） |
| `REC_HEIGHT` | `48` | 识别输入高度 |
| `REC_MAX_WIDTH` | `320` | 识别输入最大宽度 |
| `PORT` | `9899` | 监听端口 |

## 实现要点

1. **预处理只要 resize**：图片按比例缩放到高 48、宽不超过 320，归一化到 `[-1, 1]`。
   **不要**加二值化 / 降噪 / 白边——实测这些处理会让模型丢字（某 6 位验证码只剩 4 位）。
2. **CTC 解码**：`argmax` → 去 blank（索引 0）→ 去连续重复 → 按索引取词典（`chars[i-1]`）。
3. **全角归一化**：模型有时输出全角字符（`⑨`、`Ａ`），服务端统一转半角。

## 实测数据

环境：Intel i5-9500（6 核）/ 30 GB 内存 / 纯 CPU（onnxruntime）。

| 场景 | 耗时 |
|---|---|
| 单张 108×40 验证码（含 HTTP） | ~15 ms |
| 单张 320×48 纯推理 | 10~14 ms |
| batch=8 吞吐 | 126 张/秒 |

核显（Intel UHD 630，通过 ONNX Runtime + OpenVINO EP）实测：

| 任务 | CPU | 核显 |
|---|---|---|
| rec batch=1 | 10.4 ms | 28.2 ms |
| rec batch=16 | 9.6 ms | 17.4 ms |
| det 640×640 | 58.5 ms | 169.2 ms |
| det 1600×1600 | 471.2 ms | 1013.3 ms |

结论：**这类小模型在核显上全面落后 CPU**，不建议启用 GPU。核显只在计算量远超数据搬运开销的大模型（CLIP、VLM）上才可能有收益。

## 局限

- 只做**单行识别**，不含文字检测（不含 det 模型）。整页文档需要另外接入 `PP-OCRv6_small_det` + DB 后处理。
- 对**标准黑白验证码**效果好；对特殊配色、强扭曲、低分辨率的验证码，准确率取决于模型训练分布，必要时需要用站点自己的样本微调。
- 词典是 PP-OCRv6 的 18708 字符集，含中文、日文、拉丁字母与 emoji；如只需英文数字，可用 `set_ranges` 思路在解码层裁剪。

## License

MIT
