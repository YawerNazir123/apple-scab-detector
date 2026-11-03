"""Download the trained weights for apple-scab-detector.

The 55.7 MB checkpoint lives on Kaggle (public dataset
``yawarnazir/apple-scab-detector-model``) instead of in git. Run once:

    python download_model.py

It downloads the dataset zip and extracts ``best_checkpoint.pth``,
``classes.json`` and ``metrics.json`` into ``outputs/``. Stdlib only —
no Kaggle account or API key needed (the dataset is public).
"""
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DATASET = "yawarnazir/apple-scab-detector-model"
URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"
FILES = ["best_checkpoint.pth", "classes.json", "metrics.json"]


def main():
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    missing = [f for f in FILES if not (out / f).is_file()]
    if not missing:
        print("outputs/ already complete:", ", ".join(FILES))
        return
    print(f"downloading {DATASET} (~56 MB) ...")
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "model.zip"
        urllib.request.urlretrieve(URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        for f in missing:
            hits = [p for p in Path(tmp).rglob(f) if p.is_file()]
            if not hits:
                raise RuntimeError(f"{f} not found in downloaded dataset")
            shutil.copy2(hits[0], out / f)
            print("saved", out / f)
    print("done.")


if __name__ == "__main__":
    main()
