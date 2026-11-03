"""Evaluate a trained checkpoint on the validation split and print a report."""
import argparse
import json
from pathlib import Path

import torch

from src.data import LeafDataset, build_transforms
from src.train import build_model, evaluate, load_checkpoint_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--full", action="store_true",
                    help="use the whole folder as validation (no 85/15 re-split); "
                         "point --data at the dataset's valid/ folder")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_tf = build_transforms()
    val_ds = LeafDataset(args.data, split="val", transform=eval_tf,
                         resplit=not args.full)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)

    model = build_model().to(device)
    model.load_state_dict(load_checkpoint_state(args.ckpt, device))
    m = evaluate(model, loader, device)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
