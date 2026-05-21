# 2-D U-Net Training Walkthrough

This document walks you through every step to build the dataset, train the 2-D U-Net model, and evaluate results.

---

## Project File Layout

```
/Volumes/Expansion1TB/MS/
├── data/
│   ├── MSLesSeg/MSLesSeg Dataset/train/   ← source (preprocessed, 182x218x182)
│   └── ISBI2015/training/                 ← source (preprocessed, 181x217x181 → padded)
│
├── model_dataset/                         ← created by prepare_dataset.py
│   ├── train/         (90% MSLesSeg patients, all their timepoints)
│   ├── test/          (10% MSLesSeg patients, all their timepoints)
│   └── external_val/  (5 ISBI2015 subjects, zero-padded)
│
└── dummy_model_2d/
    ├── model.py            ← 2-D U-Net architecture (given)
    ├── prepare_dataset.py  ← Step 1 – build model_dataset/
    ├── dataset.py          ← TF data pipeline (2-D slices, padding)
    ├── train.py            ← Step 2 – train the model
    ├── evaluate.py         ← Step 3 – per-study Dice & IoU
    └── runs/
        └── <run_id>/
            ├── best_model.keras
            ├── training_log.csv
            ├── summary.json
            └── eval_val.json / eval_external_val.json
```

---

## Prerequisites

Activate the project virtual environment:

```bash
cd /Volumes/Expansion1TB/MS
source venv/bin/activate
```

Verify required packages are present:

```bash
pip install tensorflow nibabel numpy
```

---

## Step 1 — Build the Model Dataset

This script creates a clean, self-contained copy of the data under `model_dataset/`.

**What it does:**
- Pools **all 75 MSLesSeg patients** from two source directories:
  - `MSLesSeg Dataset/train/` — 53 patients (P1–P53), timepoint subfolders (T1, T2, T3...)
  - `MSLesSeg Dataset/test/` — 22 patients (P54–P75), flat structure (single study each)
- Groups all timepoints by **patient ID** then applies a reproducible **patient-level** `90 / 10` split (seed=42) — preventing data leakage across longitudinal scans.
- Copies FLAIR, T1, T2, and MASK for each study into `train/` or `test/`.
- Zero-pads every ISBI2015 preprocessed volume from `(181, 217, 181)` → `(182, 218, 182)` and saves into `external_val/`.

> [!NOTE]
> The `external_val/` set also contains PD modality files (`pd.nii.gz`). These are available but are **not used by the single-channel FLAIR model**. They are preserved for future multi-channel experiments.

```bash
cd /Volumes/Expansion1TB/MS/dummy_model_2d
python prepare_dataset.py
```

**Expected output (last few lines):**
```
Dataset preparation complete.
  train          : 68 patients (~105 studies)
  test           : 7  patients (~10 studies)
  external_val   : 21 studies    (5 subjects × ~4 timepoints each)
```

---

## Step 2 — Train the Model

The training script slices each 3-D volume into 2-D axial slices, pads them to `256 × 256`, stacks **FLAIR + T1w + T2w** as a 3-channel image, and feeds them to the 2-D U-Net.

### Split roles during training

| Split | Source | Role |
|---|---|---|
| `train/` | MSLesSeg 90% patients | **Training** |
| `external_val/` | ISBI2015 (5 subjects) | **Validation** — `validation_data` in `model.fit()`; used for checkpointing & early stopping each epoch |
| `test/` | MSLesSeg 10% patients | **Held-out test** — evaluated **once** after training is complete |

> [!IMPORTANT]
> The test set is never seen during training. This ensures unbiased final performance numbers.

**Key design choices in `train.py`:**

| Choice | Detail |
|---|---|
| **Input shape** | `(256, 256, 3)` — FLAIR / T1w / T2w channels |
| **Missing modality** | T1 or T2 absent → replaced by a zero tensor |
| **Loss** | Binary Cross-Entropy (stable; Dice & IoU added as metrics) |
| **Optimiser** | Adam, default LR = `1e-5` |
| **Blank-slice filtering** | 95% of lesion-free training slices dropped by default (configurable via `--skip_blank_ratio`) |
| **Validation** | ISBI2015 external set — all slices kept, no blank filtering applied |
| **Checkpointing** | Saves epoch with highest `val_dice_score` (ISBI2015 validation) |
| **LR schedule** | `fixed` by default — toggle `plateau` to add ReduceLROnPlateau |
| **Early stopping** | patience=15 on `val_dice_score` |

### Run with defaults (50 epochs, batch size 8, fixed LR = 1e-5):

```bash
python train.py
```

### Run with ReduceLROnPlateau enabled:

```bash
python train.py --lr_schedule plateau
```

### Blank-slice filtering options (`--skip_blank_ratio`):

```bash
# Default: drop 95% of lesion-free training slices
python train.py

# Keep ALL slices (no filtering):
python train.py --skip_blank_ratio 0.0

# Drop 80% of blank slices (less aggressive):
python train.py --skip_blank_ratio 0.8
```

| `--skip_blank_ratio` | Behaviour |
|---|---|
| `0.95` *(default)* | Drop 95% of training slices with no lesion voxels |
| `0.0` | Keep every slice — full class distribution |
| `0.0 – 1.0` | Any fraction is accepted |

> [!NOTE]
> Validation and test splits **always** use all slices (ratio fixed at 0.0 internally).

### Run with custom hyperparameters:

```bash
python train.py --epochs 100 --batch_size 16 --lr 1e-5 --skip_blank_ratio 0.0
python train.py --lr_schedule plateau --epochs 100 --skip_blank_ratio 0.8
```

### LR schedule options:

| `--lr_schedule` | Behaviour |
|---|---|
| `fixed` *(default)* | LR stays constant at the value given by `--lr` throughout all epochs |
| `plateau` | Adds `ReduceLROnPlateau`: halves LR when `val_dice_score` does not improve for 5 epochs; floor at `1e-7` |

### What gets printed each epoch:

```
Epoch 12/50
 loss: 0.0312 - accuracy: 0.9871 - dice_score: 0.7243 - iou_score: 0.5913
 val_loss: 0.0398 - val_accuracy: 0.9855 - val_dice_score: 0.6987 - val_iou_score: 0.5671
```

(`val_*` = ISBI2015 validation set, computed each epoch)

At the end of training the following are produced automatically:

- **Per-epoch results table** printed to terminal (Train + ISBI2015-val Dice, IoU, Loss, Accuracy)
- **`dice_curve.png`** — Dice Coefficient vs Epochs (train in blue, ISBI2015-val in red, best epoch marked)
- **`iou_curve.png`** — IoU Score vs Epochs (same format)
- **Held-out test evaluation** — MSLesSeg 10% evaluated once and printed to console
- **`summary.json`** — `val_isbi2015` (best epoch) + `test_mslesseg` (post-training) metrics
- **`training_log.csv`** — raw epoch-by-epoch CSV log

All files are saved to `runs/<run_id>/`.

---

## Step 3 — Standalone Evaluation

After training, run `evaluate.py` to get a **detailed per-study table** of Dice and IoU scores.

### Evaluate on MSLesSeg internal test set (10% patients):

```bash
python evaluate.py \
  --checkpoint runs/<run_id>/best_model.keras \
  --split test
```

### Evaluate on ISBI2015 external validation:

```bash
python evaluate.py \
  --checkpoint runs/<run_id>/best_model.keras \
  --split external_val
```

### Example output:

```
Evaluating 10 studies in: /Volumes/Expansion1TB/MS/model_dataset/val

Study                                  Dice      IoU
------------------------------------------------------
  P1_T1                              0.7421   0.5898
  P3_T2                              0.6812   0.5163
  P12_T3                             0.8104   0.6817
  ...
------------------------------------------------------
  MEAN                               0.7312   0.5791
  STD                                0.0589   0.0623
```

Detailed per-study JSON is saved to `runs/<run_id>/eval_val.json`.

---

## Metric Definitions

| Metric | Formula | Notes |
|---|---|---|
| **Dice Similarity Score** | $\frac{2 \|Y \cap \hat{Y}\|}{|Y| + |\hat{Y}|}$ | Range [0,1]; higher is better |
| **IoU (Jaccard Index)** | $\frac{\|Y \cap \hat{Y}\|}{|Y \cup \hat{Y}|}$ | Range [0,1]; stricter than Dice |

Both metrics are computed at the **volume level** (all slices of a study reconstructed before scoring) in `evaluate.py` to give clinically meaningful numbers.

---

## Interpreting Results

| Dice Range | Interpretation |
|---|---|
| > 0.80 | Excellent — clinical-grade quality |
| 0.65–0.80 | Good — acceptable for research use |
| 0.50–0.65 | Fair — model learns but under-segments |
| < 0.50 | Poor — training likely needs adjustment |

> [!TIP]
> For MS lesion segmentation, a Dice score of **≥ 0.60** on an external dataset is generally considered competitive in the literature for small-lesion scenarios.

> [!WARNING]
> The MSLesSeg dataset contains **no healthy control subjects** (all patients have confirmed lesions). If you later combine with a dataset containing controls, the blank-slice filtering ratio may need adjustment.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `OOM (out of memory)` | Reduce `--batch_size` to 4 or 2 |
| `dice_score` stuck at 0 | Check that masks are loading correctly; verify `mask.nii.gz` exists in each study folder |
| `best_model.keras` not saving | Confirm `val_dice_score` is improving; try reducing `patience` |
| ISBI zero-padding mismatch | Re-run `prepare_dataset.py`; check `model_dataset/external_val/` shapes |
