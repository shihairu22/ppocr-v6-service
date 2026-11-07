FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn onnxruntime numpy pillow

COPY ocr_app.py /app/ocr_app.py

EXPOSE 9899

CMD ["python", "ocr_app.py"]
