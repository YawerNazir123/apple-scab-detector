# apple-scab-detector

A binary image classifier that tells whether an apple leaf shows **apple scab**
(*Venturia inaequalis*) or is **healthy** — built for Kashmir, where scab is the
#1 apple disease and farmers still diagnose it by eye.

Upload a leaf photo to the API, get back `scab` / `healthy` with a confidence
score and a practical next-step note.

## Honest scope

- **Binary on purpose.** This model answers one question: scab or healthy. It
  does not claim to know black rot, cedar apple rust, or any other disease —
  training on those would be dishonest scope creep.
- **Lab photos, not orchard photos.** Training data is the PlantVillage apple
  subset (controlled lighting, plain backgrounds). It proves the pipeline; it
  does **not** prove the model works on phone photos taken in a Kashmiri
  orchard. Field validation with local photos is the real next step.
- The treatment note in the API response is informational. Always follow the
  current SKUAST-K / J&K horticulture advisory and a local officer's guidance.

## Data

PlantVillage apple subset via Kaggle
(`vipoooool/new-plant-diseases-dataset`), **Augmented** variant. The kernel
trains on the dataset's own train/valid folders (and fails loudly if they
ever resolve to the same folder — no train/val leakage):

| split | scab | healthy |
| --- | ---: | ---: |
| train | 2,016 | 2,008 |
| valid | 504 | 502 |

5,030 photos total; the augmentation pass balanced the classes (~1:1), so the
weighted sampler is near-uniform here. Only the `Apple___Apple_scab` and
`Apple___healthy` folders are used. The raw (non-augmented) PlantVillage
counts are ~630 scab + ~1,645 healthy — the augmented set is derived from
those.

## Model

ConvNeXt-Tiny fine-tune (pretrained on ImageNet), two-phase recipe:

1. **Head only**, 2 epochs, lr 3e-4 — learn the new decision boundary cheaply.
2. **Full model**, 8 epochs, lr 5e-5, cosine schedule — adapt features gently.

Weighted sampling handles the ~1:2.6 class imbalance; label smoothing 0.05;
AMP on GPU.

Trained 2026-11-02 on a Kaggle T4 GPU (~10 min total): head-only (norm + fc)
for 2 epochs, then the full model for 8 more. Best checkpoint at epoch 3 —
validation accuracy **100.0%**, scab precision **100.0%**, recall **100.0%**,
F1 **100.0%** (504 scab + 502 healthy validation photos). Binary is much
easier than 700-way bird classification and PlantVillage's lab photos are
consistent, so near-perfect on the validation split is expected — it proves
the pipeline end to end, not orchard performance.

Reported metrics (validation split): accuracy, scab precision/recall/F1 —
saved to `outputs/metrics.json` with the best checkpoint. The checkpoint's
weights are stored in float16 (55.7 MB) and hosted on Kaggle as the public
dataset
[yawarnazir/apple-scab-detector-model](https://www.kaggle.com/datasets/yawarnazir/apple-scab-detector-model)
instead of in git — run `python download_model.py` to fetch
`best_checkpoint.pth` (+ `classes.json`, `metrics.json`) into `outputs/`.
The training kernel re-loaded the shipped fp16 file and re-evaluated it
before finishing (still 100.0%), confirming the shipped file works end to
end.

## Project layout

```
apple-scab-detector/
├── src/
│   ├── __init__.py   # class constants, dataset URL
│   ├── data.py       # dataset, transforms, weighted sampler
│   ├── train.py      # two-phase fine-tune
│   ├── evaluate.py   # metrics report for a checkpoint
│   ├── predict.py    # single-image CLI inference
│   └── api.py        # FastAPI service (POST /predict)
├── tests/            # pytest suite (API schema, data split, transforms)
├── kaggle-kernel/    # self-contained training kernel for Kaggle GPU
├── download_model.py # fetch the trained weights from Kaggle into outputs/
├── outputs/         # classes.json + metrics.json (+ best_checkpoint.pth after download)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Run it

```bash
# 0. fetch the trained weights (once):
python download_model.py
# 1. to retrain: download the Kaggle dataset (vipoooool/new-plant-diseases-dataset).
#    The Augmented train/ and valid/ folders hold the class dirs.
# 2. train — the same recipe the shipped checkpoint was trained with:
AUG="<dataset>/New Plant Diseases Dataset(Augmented)"
python -m src.train --data "$AUG/train" --val-data "$AUG/valid" --out outputs
#    (omit --val-data to train on any single folder with an 85/15 re-split)
# 3. evaluate
python -m src.evaluate --data "$AUG/valid" --full --ckpt outputs/best_checkpoint.pth
# 4. serve
uvicorn src.api:app --port 8000
# 5. predict
curl -F "file=@leaf.jpg" http://localhost:8000/predict
```

Docker:

```bash
python download_model.py    # fetch the trained weights first
docker compose up --build
```

Note: the first `/predict` call downloads the ConvNeXt-Tiny ImageNet weights
(~110 MB) into the timm cache, so the container needs internet on first use.

Tests:

```bash
pytest -q
```

## Training on Kaggle (free GPU)

The kernel in `kaggle-kernel/` attaches the public
`vipoooool/new-plant-diseases-dataset` dataset and runs the same two-phase
recipe on a T4 (took ~10 min), saving `best_checkpoint.pth` + `metrics.json`
as kernel outputs. It uses the dataset's own Augmented train/valid folders
(4,024 / 1,006 images) and fails loudly if they ever resolve to the same
folder, so there is no train/val leakage. Push with:

```bash
kaggle kernels push -p kaggle-kernel/
```

## Why this project

Kashmir's cool, wet springs are ideal for apple scab, and it causes the
heaviest yield losses of any apple disease in the valley. Diagnosis is still
visual and subjective — slow, and it misses early infections. A phone-photo
classifier is a small, honest step toward catching it earlier.
