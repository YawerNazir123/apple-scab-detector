"""
FastAPI service for apple-scab-detector.

POST /predict  (multipart, field "file" = leaf photo)
  -> {"label": "scab"|"healthy", "confidence": float,
      "probs": {...}, "advice": "..."}

Run:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src import CLASS_NAMES
from src.data import build_transforms
from src.train import build_model, load_checkpoint_state

CKPT = os.environ.get("SCAB_CKPT", "outputs/best_checkpoint.pth")

ADVICE = {
    "scab": ("Apple scab detected (Venturia inaequalis). In Kashmir this is the "
             "most damaging apple disease: lesions spread fast in cool, wet spring "
             "weather. Check the current SKUAST-K / J&K horticulture spray schedule "
             "for your area and consult a local horticulture officer before acting. "
             "This is an informational model output, not agronomic advice."),
    "healthy": ("No scab detected in this leaf. Keep to the regular orchard "
                "monitoring routine, especially after wet spells in spring."),
}

app = FastAPI(title="apple-scab-detector",
              description="Binary apple-scab vs healthy leaf classifier")

_model = None
_device = None
_eval_tf = None


def get_model():
    global _model, _device, _eval_tf
    if _model is None:
        if not Path(CKPT).is_file():
            raise RuntimeError(
                f"checkpoint not found: {CKPT} — run `python download_model.py` "
                "to fetch the trained weights")
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = build_model().to(_device)
        _model.load_state_dict(load_checkpoint_state(CKPT, _device))
        _model.eval()
        _, _eval_tf = build_transforms()
    return _model, _device, _eval_tf


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": Path(CKPT).name,
            "checkpoint_present": Path(CKPT).is_file()}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(400, "upload a JPEG/PNG/WebP leaf photo")
    try:
        get_model()
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "could not decode image")
    x = _eval_tf(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        probs = torch.softmax(_model(x), dim=1)[0]
    idx = int(probs.argmax())
    label = CLASS_NAMES[idx]
    return JSONResponse({
        "label": label,
        "confidence": round(float(probs[idx]), 4),
        "probs": {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(2)},
        "advice": ADVICE[label],
    })


@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse("static/index.html")


# Mount static files AFTER API routes so /predict, /health etc. are not
# shadowed by the static file handler.
app.mount("/static", StaticFiles(directory="static"), name="static")
