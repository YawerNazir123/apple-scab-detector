"""Download the trained weights for apple-scab-detector.

The 54 MB checkpoint lives on Hugging Face (public model repo
``yawarnazir/apple-scab-detector-model``) instead of in git. Run once:

    python download_model.py

It downloads ``best_checkpoint.pth``, ``classes.json`` and
``metrics.json`` into ``outputs/``. Stdlib only — no Hugging Face
account or API token needed (the repo is public).
"""
import urllib.request
from pathlib import Path

HF_REPO = "yawarnazir/apple-scab-detector-model"
BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"
FILES = ["best_checkpoint.pth", "classes.json", "metrics.json"]


def main():
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    missing = [f for f in FILES if not (out / f).is_file()]
    if not missing:
        print("outputs/ already complete:", ", ".join(FILES))
        return
    print(f"downloading {len(missing)} file(s) from Hugging Face repo {HF_REPO} ...")
    for f in missing:
        url = f"{BASE_URL}/{f}"
        dest = out / f
        print(f"downloading {f} ...")
        urllib.request.urlretrieve(url, dest)
        print("saved", dest)
    print("done.")


if __name__ == "__main__":
    main()
