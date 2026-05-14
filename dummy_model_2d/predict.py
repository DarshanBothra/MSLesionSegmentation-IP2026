"""
predict.py
----------
Load a trained 2-D U-Net checkpoint and run inference on a study directory
containing flair.nii.gz (required), t1.nii.gz, t2.nii.gz, and mask.nii.gz.

Two modes of operation
----------------------
  1. --study_dir  : point to a folder with flair/t1/t2/mask NIfTI files.
                    Produces a 6-panel comparison figure (FLAIR, T1w, T2w,
                    Ground Truth, Predicted Mask, Overlay) for a
                    representative axial slice.

  2. --input      : (legacy) single .nii.gz treated as FLAIR-only.
                    Generates a predicted mask volume saved to --output.

Usage
-----
  # Study directory mode (recommended):
  python predict.py \\
      --study_dir /path/to/model_dataset/train/P1_T1 \\
      --model     runs/20260514_084748/best_model.h5 \\
      --output    prediction_P1_T1.png

  # Single-file legacy mode:
  python predict.py \\
      --input  /path/to/flair.nii.gz \\
      --model  runs/20260514_084748/best_model.h5 \\
      --output predicted_mask.nii.gz
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Padding / un-padding helpers
# ─────────────────────────────────────────────────────────────────────────────

TARGET_H, TARGET_W = 256, 256


def centre_pad_2d(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Zero-pad a 2-D array to (th, tw), centred."""
    h, w = arr.shape
    pad_h = th - h
    pad_w = tw - w
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left
    return np.pad(arr, ((top, bottom), (left, right)), mode="constant")


def unpad_2d(arr: np.ndarray, original_h: int, original_w: int) -> np.ndarray:
    """Unpad a 2-D array to original size, assuming it was centre-padded."""
    h, w = arr.shape
    pad_h = h - original_h
    pad_w = w - original_w
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left

    h_end = h - bottom if bottom > 0 else h
    w_end = w - right  if right  > 0 else w
    return arr[top:h_end, left:w_end]


# ─────────────────────────────────────────────────────────────────────────────
# Volume loader
# ─────────────────────────────────────────────────────────────────────────────

def load_nii(path):
    """Load a NIfTI file and return (data_float32, nib_img)."""
    if not os.path.exists(path):
        return None, None
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32), img


# ─────────────────────────────────────────────────────────────────────────────
# Predict on a full study directory  (FLAIR + T1 + T2 + optional mask)
# ─────────────────────────────────────────────────────────────────────────────

def predict_study(study_dir: str, model_path: str, output_path: str):
    """
    Run inference on a study folder and save a 6-panel comparison figure
    for the axial slice with the most lesion voxels.
    """
    # ── Load modalities ───────────────────────────────────────────────────────
    flair, flair_nii = load_nii(os.path.join(study_dir, "flair.nii.gz"))
    t1,    _         = load_nii(os.path.join(study_dir, "t1.nii.gz"))
    t2,    _         = load_nii(os.path.join(study_dir, "t2.nii.gz"))
    mask,  _         = load_nii(os.path.join(study_dir, "mask.nii.gz"))

    if flair is None:
        print(f"ERROR: flair.nii.gz not found in {study_dir}")
        sys.exit(1)

    H, W, D = flair.shape
    if t1 is None:
        t1 = np.zeros_like(flair)
        print("  [INFO] t1.nii.gz missing — filled with zeros.")
    if t2 is None:
        t2 = np.zeros_like(flair)
        print("  [INFO] t2.nii.gz missing — filled with zeros.")

    has_gt = mask is not None
    if has_gt:
        mask = (mask > 0.5).astype(np.float32)
    else:
        print("  [INFO] mask.nii.gz missing — ground truth will not be shown.")

    # ── Build 3-channel input slices ──────────────────────────────────────────
    print(f"Volume shape: {H}×{W}×{D}   |   Preparing slices ...")
    imgs_list = []
    for z in range(D):
        f_s  = centre_pad_2d(flair[:, :, z], TARGET_H, TARGET_W)
        t1_s = centre_pad_2d(t1[:, :, z],    TARGET_H, TARGET_W)
        t2_s = centre_pad_2d(t2[:, :, z],    TARGET_H, TARGET_W)
        imgs_list.append(np.stack([f_s, t1_s, t2_s], axis=-1))

    X = np.array(imgs_list, dtype=np.float32)       # (D, 256, 256, 3)

    # ── Load model & predict ──────────────────────────────────────────────────
    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(model_path, compile=False)

    print("Running inference ...")
    preds = model.predict(X, batch_size=8, verbose=1)       # (D, 256, 256, 1)
    preds_bin = (preds > 0.5).astype(np.float32)[..., 0]    # (D, 256, 256)

    # Un-pad predictions back to original spatial dims
    pred_vol = np.zeros((H, W, D), dtype=np.float32)
    for z in range(D):
        pred_vol[:, :, z] = unpad_2d(preds_bin[z], H, W)

    # ── Pick the most informative slice ───────────────────────────────────────
    # Use the slice with the most lesion voxels (from GT if available, else prediction)
    ref_vol = mask if has_gt else pred_vol
    lesion_per_slice = ref_vol.sum(axis=(0, 1))             # (D,)
    best_z = int(np.argmax(lesion_per_slice))
    print(f"Selected axial slice z={best_z} (most lesion voxels)")

    # ── Plot 6-panel figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Row 1: FLAIR, T1w, T2w
    panels_row1 = [
        (flair[:, :, best_z], "FLAIR",  "gray"),
        (t1[:, :, best_z],    "T1w",    "gray"),
        (t2[:, :, best_z],    "T2w",    "gray"),
    ]
    for ax, (data, title, cmap) in zip(axes[0], panels_row1):
        ax.imshow(np.rot90(data), cmap=cmap, aspect="equal")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

    # Row 2: Ground Truth, Predicted Mask, Overlay
    gt_slice   = mask[:, :, best_z] if has_gt else np.zeros((H, W))
    pred_slice = pred_vol[:, :, best_z]
    flair_slice = flair[:, :, best_z]

    # Ground Truth
    axes[1][0].imshow(np.rot90(gt_slice), cmap="Reds", aspect="equal")
    axes[1][0].set_title("Ground Truth Mask" if has_gt else "GT (unavailable)",
                         fontsize=14, fontweight="bold")
    axes[1][0].axis("off")

    # Predicted Mask
    axes[1][1].imshow(np.rot90(pred_slice), cmap="Reds", aspect="equal")
    axes[1][1].set_title("Predicted Mask", fontsize=14, fontweight="bold")
    axes[1][1].axis("off")

    # Overlay: FLAIR + GT (green) + Prediction (red)
    flair_norm = flair_slice / (flair_slice.max() + 1e-7)
    overlay = np.stack([flair_norm, flair_norm, flair_norm], axis=-1)  # (H, W, 3)
    if has_gt:
        overlay[..., 1] = np.clip(overlay[..., 1] + gt_slice * 0.4, 0, 1)   # green = GT
    overlay[..., 0] = np.clip(overlay[..., 0] + pred_slice * 0.4, 0, 1)      # red   = pred
    axes[1][2].imshow(np.rot90(overlay), aspect="equal")
    overlay_title = "Overlay (Green=GT, Red=Pred)" if has_gt else "Overlay (Red=Pred)"
    axes[1][2].set_title(overlay_title, fontsize=14, fontweight="bold")
    axes[1][2].axis("off")

    # ── Dice / IoU annotation ─────────────────────────────────────────────────
    if has_gt:
        inter  = (mask * pred_vol).sum()
        union_ = mask.sum() + pred_vol.sum()
        dice   = (2.0 * inter + 1e-7) / (union_ + 1e-7)
        iou    = (inter + 1e-7) / (union_ - inter + 1e-7)
        fig.suptitle(
            f"Study: {os.path.basename(study_dir)}   |   "
            f"Slice z={best_z}   |   "
            f"Volume Dice={dice:.4f}   IoU={iou:.4f}",
            fontsize=15, fontweight="bold", y=0.98,
        )
    else:
        fig.suptitle(
            f"Study: {os.path.basename(study_dir)}   |   Slice z={best_z}",
            fontsize=15, fontweight="bold", y=0.98,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Comparison figure saved to: {output_path}")

    # Also save predicted mask as NIfTI next to the figure
    mask_nii_path = output_path.replace(".png", "_pred_mask.nii.gz")
    out_nii = nib.Nifti1Image(pred_vol, flair_nii.affine, flair_nii.header)
    nib.save(out_nii, mask_nii_path)
    print(f"Predicted mask volume saved to: {mask_nii_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Legacy: single-file (FLAIR only) prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(input_path: str, model_path: str, output_path: str):
    """Original single-file mode: treat input as FLAIR, zero-fill T1/T2."""
    print(f"Loading input image from {input_path}...")
    img_nii = nib.load(input_path)
    img_data = img_nii.get_fdata(dtype=np.float32)

    H, W, D = img_data.shape
    print(f"Original shape: {H}x{W}x{D}")
    print("Preparing slices...")

    imgs_list = []
    for z in range(D):
        f_s  = centre_pad_2d(img_data[:, :, z], TARGET_H, TARGET_W)
        t1_s = np.zeros_like(f_s)
        t2_s = np.zeros_like(f_s)
        imgs_list.append(np.stack([f_s, t1_s, t2_s], axis=-1))

    X = np.array(imgs_list, dtype=np.float32)

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)

    print("Running inference...")
    preds = model.predict(X, batch_size=8, verbose=1)
    preds_bin = (preds > 0.5).astype(np.float32)[..., 0]

    print("Reconstructing volume...")
    out_mask = np.zeros((H, W, D), dtype=np.float32)
    for z in range(D):
        out_mask[:, :, z] = unpad_2d(preds_bin[z], H, W)

    print(f"Saving predicted mask to {output_path}...")
    out_nii = nib.Nifti1Image(out_mask, img_nii.affine, img_nii.header)
    nib.save(out_nii, output_path)
    print("Done!")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict MS-lesion mask and visualise results")
    parser.add_argument("--study_dir", default=None,
                        help="Path to a study folder (flair/t1/t2/mask .nii.gz)")
    parser.add_argument("--input", default=None,
                        help="(Legacy) Path to a single .nii.gz FLAIR image")
    parser.add_argument("--model", required=True,
                        help="Path to trained model checkpoint (.h5 or .keras)")
    parser.add_argument("--output", required=True,
                        help="Output path (.png for study mode, .nii.gz for legacy)")
    args = parser.parse_args()

    if args.study_dir:
        predict_study(args.study_dir, args.model, args.output)
    elif args.input:
        predict_single(args.input, args.model, args.output)
    else:
        parser.error("Provide either --study_dir or --input")
