"""Single-image inference: leaf photo -> scab / healthy + confidence."""
import argparse
from pathlib import Path

import torch
from PIL import Image

from src import CLASS_NAMES
from src.data import build_transforms
from src.train import build_model, load_checkpoint_state


def load_model(ckpt_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(load_checkpoint_state(ckpt_path, device))
    model.eval()
    return model, device


def predict_image(model, device, image_path):
    _, eval_tf = build_transforms()
    img = Image.open(image_path).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return {"label": CLASS_NAMES[idx], "confidence": float(probs[idx]),
            "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(2)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    model, device = load_model(args.ckpt)
    import json
    print(json.dumps(predict_image(model, device, args.image), indent=2))


if __name__ == "__main__":
    main()
