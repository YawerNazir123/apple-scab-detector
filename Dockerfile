FROM python:3.11-slim

WORKDIR /app

# torch CPU wheel keeps the image small; override for GPU builds.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY static/ ./static/

ENV SCAB_CKPT=/app/outputs/best_checkpoint.pth
EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
