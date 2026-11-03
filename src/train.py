"""
Train the apple-scab binary classifier.

Two-phase fine-tune of ConvNeXt-Tiny (the same proven recipe as the bird-ID
model): freeze the backbone and train the head first, then unfreeze and train
everything with a small learning rate.

Usage (local):
    python -m src.train --data data/plantvillage-apple --epochs-full 8

On Kaggle the standalone kernel script (kaggle-kernel/train_scab.py) is used
instead, pointing at /kaggle/input.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

import timm

from src.data import make_loaders, IMG_SIZE


def build_model(num_classes=2):
    model = timm.create_model("convnext_tiny", pretrained=True, num_classes=num_classes)
    return model


def save_checkpoint(model, path, metrics):
    """Save best weights + metrics.

    Weights are stored in float16: ConvNeXt-Tiny is ~112 MB in float32,
    which exceeds GitHub's 100 MB per-file limit. float16 halves the file
    with no meaningful accuracy change. Load with load_checkpoint_state().
    """
    sd16 = {k: v.half() for k, v in model.state_dict().items()}
    torch.save({"model": sd16, "metrics": metrics}, path)


def load_checkpoint_state(ckpt_path, device):
    """Load checkpoint weights (float16 or float32) as a float32 state dict."""
    sd = torch.load(ckpt_path, map_location=device)["model"]
    return {k: v.to(torch.float32) for k, v in sd.items()}


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    tp = fp = fn = 0  # scab = class 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            tp += ((pred == 0) & (y == 0)).sum().item()
            fp += ((pred == 0) & (y == 1)).sum().item()
            fn += ((pred == 1) & (y == 0)).sum().item()
    acc = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"acc": acc, "precision_scab": precision, "recall_scab": recall, "f1_scab": f1}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} img={IMG_SIZE}")

    train_loader, val_loader, _ = make_loaders(
        args.data, val_root=args.val_data, batch=args.batch,
        workers=args.workers, seed=args.seed)
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    model = build_model().to(device)
    scaler = GradScaler(enabled=device.type == "cuda")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = {"acc": 0.0}

    def run_phase(epochs, lr, freeze_backbone, start_epoch):
        if freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
            if hasattr(model, "head"):
                head_params = model.head.parameters()  # norm + fc, as in the Kaggle kernel
            else:
                head_params = model.get_classifier().parameters()
            for p in head_params:
                p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = True
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for ep in range(epochs):
            model.train()
            t0 = time.time()
            tot_loss, n = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                with autocast(enabled=device.type == "cuda"):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits, y, label_smoothing=0.05)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot_loss += loss.item() * y.numel()
                n += y.numel()
            sched.step()
            m = evaluate(model, val_loader, device)
            line = (f"epoch {start_epoch + ep + 1:>2} "
                    f"loss={tot_loss / max(n, 1):.4f} "
                    f"val_acc={m['acc']:.4f} val_f1_scab={m['f1_scab']:.4f} "
                    f"({time.time() - t0:.0f}s)")
            print(line, flush=True)
            if m["acc"] > best["acc"]:
                best = {"acc": m["acc"], **m, "epoch": start_epoch + ep + 1}
                save_checkpoint(model, out / "best_checkpoint.pth", best)
                print(f"  -> new best, saved (acc={m['acc']:.4f})", flush=True)
        return epochs

    e = run_phase(args.epochs_head, args.lr_head, freeze_backbone=True, start_epoch=0)
    run_phase(args.epochs_full, args.lr_full, freeze_backbone=False, start_epoch=e)

    (out / "metrics.json").write_text(json.dumps(best, indent=2))
    print("BEST:", json.dumps(best, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with Apple___Apple_scab/ and Apple___healthy/")
    ap.add_argument("--val-data", default=None,
                    help="optional separate validation folder with the two class "
                         "dirs; when given, --data is used whole for training "
                         "(no 85/15 re-split). Use this to reproduce the "
                         "Kaggle-trained model exactly.")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs-head", type=int, default=2)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--epochs-full", type=int, default=8)
    ap.add_argument("--lr-full", type=float, default=5e-5)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
