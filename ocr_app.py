#!/usr/bin/env python3
"""独立 OCR 服务（多模型路由版）：
- 默认 PP-OCRv6 small（通用验证码 / 短文本）
- 请求体带 "model": "pthome" 时走专用 CRNN（针对 pthome 验证码微调）
接口与旧容器一致：GET /captcha/health、POST /captcha/base64
"""
import base64
import io
import json
import os
import time

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

REC_H = int(os.environ.get("REC_HEIGHT", "48"))
REC_MAX_W = int(os.environ.get("REC_MAX_WIDTH", "320"))
CRNN_PATH = os.environ.get("CRNN_PATH", "/app/models/crnn_best.onnx")
CRNN_CHARSET = os.environ.get("CRNN_CHARSET", "/app/models/charset.json")

_app = FastAPI(title="PP-OCRv6 OCR Service")


class _Engine:
    def __init__(self, model_path, dict_path=None, charset_path=None):
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.chars = None
        if dict_path and os.path.exists(dict_path):
            self.chars = [l.rstrip("\n") for l in open(dict_path, encoding="utf-8")]
        elif charset_path and os.path.exists(charset_path):
            self.chars = json.load(open(charset_path, encoding="utf-8"))["charset"]


V6 = _Engine(os.environ.get("MODEL_PATH", "/app/models/rec.onnx"),
             dict_path=os.environ.get("DICT_PATH", "/app/models/dict.txt"))
CRNN = None
if os.path.exists(CRNN_PATH):
    try:
        CRNN = _Engine(CRNN_PATH, charset_path=CRNN_CHARSET)
    except Exception as e:
        print("CRNN 加载失败:", e, flush=True)

U2M = None
if os.path.exists("/app/models/u2_crnn_best.onnx"):
    try:
        U2M = _Engine("/app/models/u2_crnn_best.onnx",
                      charset_path="/app/models/u2_charset.json")
        print("U2 CRNN 模型已加载", flush=True)
    except Exception as e:
        print("U2 模型加载失败:", e, flush=True)

_CIRCLED = {chr(0x2460 + i): str(i + 1) for i in range(20)}


def _normalize(text: str) -> str:
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


def _ctc(logits, chars):
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits.reshape(-1, logits.shape[-1])  # 任意前导维 → (W, NC)
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
    return _normalize("".join(text)), conf


def _prep(im):
    w, h = im.size
    nw = max(16, min(REC_MAX_W, int(round(w * REC_H / max(1, h)))))
    im2 = im.resize((nw, REC_H), Image.LANCZOS)
    a = np.asarray(im2).astype(np.float32) / 255.0
    a = (a - 0.5) / 0.5
    return a.transpose(2, 0, 1)[None, ...]


def _prep_crnn(im):
    """pthome 专用 CRNN：固定拉伸 320×48（与训练一致）。"""
    im2 = im.resize((320, 48), Image.LANCZOS)
    a = np.asarray(im2).astype(np.float32) / 255.0
    a = (a - 0.5) / 0.5
    return a.transpose(2, 0, 1)[None, ...]


class Req(BaseModel):
    base64_img: str
    preprocess: bool = True
    model: str = "v6"   # v6 | pthome


@_app.get("/captcha/health")
def health():
    return {
        "status": "ok",
        "ocr_service_initialized": True,
        "models": ["v6"] + (["pthome"] if CRNN else []) + (["u2"] if U2M else []),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@_app.post("/captcha/base64")
def ocr(req: Req):
    t0 = time.time()
    try:
        raw = base64.b64decode(req.base64_img)
        im = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解析失败: {e}")

    if req.model == "pthome":
        eng = CRNN or V6
    elif req.model == "u2":
        eng = U2M or V6
    else:
        eng = V6

    try:
        x = _prep_crnn(im) if (req.model in ("pthome", "u2") and eng is not V6) else _prep(im)
        out = eng.sess.run(None, {eng.inp: x})[0]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"推理失败: {e}")

    text, conf = _ctc(out, eng.chars or [])
    return {
        "result": text,
        "confidence": round(conf, 4),
        "model": req.model if (req.model == "pthome" and CRNN) else "v6",
        "processing_time": round(time.time() - t0, 4),
    }


if __name__ == "__main__":
    uvicorn.run(_app, host="0.0.0.0", port=int(os.environ.get("PORT", "9899")))
