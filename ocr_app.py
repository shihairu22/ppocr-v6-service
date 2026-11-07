#!/usr/bin/env python3
"""独立 OCR 服务（PP-OCRv6 small，CPU）：实现与旧容器一致的接口。
GET  /captcha/health   -> {"status":"ok","ocr_service_initialized":true}
POST /captcha/base64   -> {"base64_img": "...", "preprocess": true} -> {"result","confidence","processing_time"}
"""
import base64
import io
import os
import time

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

MODEL = os.environ.get("MODEL_PATH", "/app/models/rec.onnx")
DICT = os.environ.get("DICT_PATH", "/app/models/dict.txt")
REC_H = int(os.environ.get("REC_HEIGHT", "48"))
REC_MAX_W = int(os.environ.get("REC_MAX_WIDTH", "320"))

chars = [line.rstrip("\n") for line in open(DICT, encoding="utf-8")]
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name

app = FastAPI(title="PP-OCRv6 OCR Service")


class Req(BaseModel):
    base64_img: str
    preprocess: bool = True


def ctc_decode(out: np.ndarray):
    """out: [1, T, C]；C = len(chars)+2（blank=0，末尾 space）"""
    logits = out[0]
    idx = logits.argmax(axis=-1)
    probs = logits.max(axis=-1)
    text, confs, prev = [], [], -1
    for t, i in enumerate(idx):
        i = int(i)
        if i != 0 and i != prev and 1 <= i <= len(chars):
            text.append(chars[i - 1])
            confs.append(float(probs[t]))
        prev = i
    conf = float(np.mean(confs)) if confs else 0.0
    return normalize("".join(text)), conf


_CIRCLED = {chr(0x2460 + i): str(i + 1) for i in range(20)}


def normalize(text: str) -> str:
    """全角 ASCII / 全角空格 / 带圈数字 → 半角，验证码只可能是半角字符。"""
    out = []
    for ch in text:
        code = ord(ch)
        if 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            out.append(" ")
        elif ch in _CIRCLED:
            out.append(_CIRCLED[ch])
        else:
            out.append(ch)
    return "".join(out)


@app.get("/captcha/health")
def health():
    return {
        "status": "ok",
        "ocr_service_initialized": True,
        "model": os.path.basename(MODEL),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@app.post("/captcha/base64")
def ocr(req: Req):
    t0 = time.time()
    try:
        raw = base64.b64decode(req.base64_img)
        im = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解析失败: {e}")

    w, h = im.size
    nw = max(16, min(REC_MAX_W, int(round(w * REC_H / max(1, h)))))
    im2 = im.resize((nw, REC_H), Image.LANCZOS)
    a = np.asarray(im2).astype(np.float32) / 255.0
    a = (a - 0.5) / 0.5
    x = a.transpose(2, 0, 1)[None, ...]

    try:
        out = sess.run(None, {inp_name: x})[0]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"推理失败: {e}")

    text, conf = ctc_decode(out)
    return {
        "result": text,
        "confidence": round(conf, 4),
        "processing_time": round(time.time() - t0, 4),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "9899")))
