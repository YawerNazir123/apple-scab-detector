"""
Shared data utilities for apple-scab-detector.

Binary problem: Apple___Apple_scab vs Apple___healthy from the PlantVillage
family of datasets. We deliberately exclude black rot and cedar apple rust:
they are not the target here and training on them would pretend the model
knows diseases it does not. See README "Honest scope".
"""
from pathlib import Path

CLASSES = ["Apple___Apple_scab", "Apple___healthy"]
CLASS_NAMES = ["scab", "healthy"]

DATA_URL = "https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset"


def find_data_root(candidates=("/kaggle/input", "data")):
    """Locate the dataset root that contains the class folders."""
    for base in candidates:
        base = Path(base)
        if not base.is_dir():
            continue
        for root in [base] + [p for p in base.rglob("*") if p.is_dir() and len(p.parts) <= len(Path(base).parts) + 3]:
            if all((root / c).is_dir() for c in CLASSES):
                return root
    return None


def count_images(root):
    root = Path(root)
    return {
        c: sum(1 for p in (root / c).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for c in CLASSES
    }
