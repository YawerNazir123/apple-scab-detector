import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import src.api as api
from src import CLASSES
from src.data import build_transforms, LeafDataset


def make_jpeg(color=(120, 180, 90)):
    img = Image.fromarray(np.full((300, 300, 3), color, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_class_constants():
    assert CLASSES == ["Apple___Apple_scab", "Apple___healthy"]


def test_transforms_roundtrip():
    train_tf, eval_tf = build_transforms()
    buf = make_jpeg()
    img = Image.open(buf)
    assert train_tf(img).shape == (3, 224, 224)
    assert eval_tf(img).shape == (3, 224, 224)


def test_predict_endpoint_shape(monkeypatch, tmp_path):
    """API returns the documented schema even with a stubbed model."""
    import torch
    import torch.nn as nn

    class Stub(nn.Module):
        def forward(self, x):
            return torch.tensor([[2.0, 0.5]])  # scab wins

    monkeypatch.setattr(api, "_model", Stub())
    monkeypatch.setattr(api, "_device", torch.device("cpu"))
    _, eval_tf = build_transforms()
    monkeypatch.setattr(api, "_eval_tf", eval_tf)

    client = TestClient(api.app)
    resp = client.post("/predict", files={"file": ("leaf.jpg", make_jpeg(), "image/jpeg")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "scab"
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probs"]) == {"scab", "healthy"}
    assert "advice" in body and len(body["advice"]) > 20


def test_predict_rejects_non_image():
    client = TestClient(api.app)
    resp = client.post("/predict", files={"file": ("x.txt", io.BytesIO(b"nope"), "text/plain")})
    assert resp.status_code == 400


def test_health_endpoint():
    client = TestClient(api.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_leaf_dataset_split(tmp_path):
    # synthetic two-class folder, deterministic split
    for cls in CLASSES:
        d = tmp_path / cls
        d.mkdir()
        for i in range(10):
            Image.fromarray(np.full((64, 64, 3), i, dtype=np.uint8)).save(d / f"{i}.jpg")
    _, eval_tf = build_transforms()
    tr = LeafDataset(tmp_path, split="train", transform=eval_tf)
    va = LeafDataset(tmp_path, split="val", transform=eval_tf)
    assert len(tr) + len(va) == 20
    assert len(tr) == 17 and len(va) == 3
    x, y = tr[0]
    assert x.shape == (3, 224, 224) and y in (0, 1)
