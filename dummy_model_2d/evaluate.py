"""
evaluate.py
-----------
Standalone evaluation script.  Loads the best model checkpoint from a
training run and reports per-study and aggregate Dice and IoU scores on
any of the three splits: val, external_val, or a custom directory.

Usage
-----
  # Evaluate on internal val set (MSLesSeg 10%)
  python evaluate.py --checkpoint runs/<run_id>/best_model.keras --split val

  # Evaluate on ISBI2015 external validation
  python evaluate.py --checkpoint runs/<run_id>/best_model.keras --split external_val

  # Evaluate on a custom directory
  python evaluate.py --checkpoint runs/<run_id>/best_model.keras \
                     --split_dir /path/to/split
"""

import os
import sys
import json
import argparse

import numpy as np
import nibabel as nib
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from dataset import extract_slices, TARGET_H, TARGET_W

# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = "/Volumes/Expansion1TB/MS/model_dataset"
THRESHOLD    = 0.5      # binarisation threshold


# ─────────────────────────────────────────────────────────────────────────────
# Metric functions (numpy, per-study)
# ─────────────────────────────────────────────────────────────────────────────

def dice_score(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float:
    inter = (y_true * y_pred_bin).sum()
    union = y_true.sum() + y_pred_bin.sum()
    if union == 0:
        return 1.0          # both empty -> perfect agreement
    return float((2.0 * inter + 1e-7) / (union + 1e-7))


def iou_score(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float:
    inter = (y_true * y_pred_bin).sum()
    union = y_true.sum() + y_pred_bin.sum() - inter
    if union == 0:
        return 1.0
    return float((inter + 1e-7) / (union + 1e-7))


# ─────────────────────────────────────────────────────────────────────────────
# Per-study evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_study(model: tf.keras.Model, study_dir: str):
    """
    Run inference on ALL slices of a study (no blank-slice skipping).
    Returns (dice, iou) for the entire 3-D volume reconstructed from
    per-slice predictions.
    """
    imgs, masks = extract_slices(study_dir, skip_blank_ratio=0.0,
                                 is_train=False)
    if imgs is None:
        print(f"  [WARN] No data in {study_dir}, skipping.")
        return None, None

    preds = model.predict(imgs, batch_size=8, verbose=0)   # (N,256,256,1)
    preds_bin = (preds > THRESHOLD).astype(np.float32)

    d = dice_score(masks, preds_bin)
    i = iou_score(masks, preds_bin)
    return d, i


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(checkpoint: str, split_dir: str):
    print(f"\nLoading model from: {checkpoint}")
    model = tf.keras.models.load_model(checkpoint, compile=False)

    studies = sorted([
        s for s in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, s))
    ])
    print(f"Evaluating {len(studies)} studies in: {split_dir}\n")
    print(f"{'Study':<35} {'Dice':>8} {'IoU':>8}")
    print("-" * 54)

    all_dice = []
    all_iou  = []
    results  = {}

    for study_id in studies:
        study_path = os.path.join(split_dir, study_id)
        d, i = evaluate_study(model, study_path)
        if d is None:
            continue
        all_dice.append(d)
        all_iou.append(i)
        results[study_id] = {"dice": round(d, 4), "iou": round(i, 4)}
        print(f"  {study_id:<33} {d:>8.4f} {i:>8.4f}")

    print("-" * 54)
    mean_dice = float(np.mean(all_dice)) if all_dice else 0.0
    mean_iou  = float(np.mean(all_iou))  if all_iou  else 0.0
    std_dice  = float(np.std(all_dice))  if all_dice else 0.0
    std_iou   = float(np.std(all_iou))   if all_iou  else 0.0

    print(f"  {'MEAN':<33} {mean_dice:>8.4f} {mean_iou:>8.4f}")
    print(f"  {'STD':<33} {std_dice:>8.4f} {std_iou:>8.4f}")

    summary = {
        "checkpoint": checkpoint,
        "split_dir":  split_dir,
        "n_studies":  len(all_dice),
        "mean_dice":  round(mean_dice, 4),
        "std_dice":   round(std_dice,  4),
        "mean_iou":   round(mean_iou,  4),
        "std_iou":    round(std_iou,   4),
        "per_study":  results,
    }

    out_path = os.path.join(
        os.path.dirname(checkpoint),
        f"eval_{os.path.basename(split_dir)}.json"
    )
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed results saved to: {out_path}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate 2-D U-Net and report Dice + IoU per study")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .keras model checkpoint")
    parser.add_argument("--split",
                        choices=["test", "external_val"],
                        default=None,
                        help="Named split under DATASET_ROOT")
    parser.add_argument("--split_dir", default=None,
                        help="Explicit path to a split directory")
    args = parser.parse_args()

    if args.split_dir:
        split_dir = args.split_dir
    elif args.split:
        split_dir = os.path.join(DATASET_ROOT, args.split)
    else:
        raise ValueError("Provide either --split or --split_dir")

    evaluate(checkpoint=args.checkpoint, split_dir=split_dir)
