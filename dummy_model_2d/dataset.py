"""
dataset.py
----------
TensorFlow/Keras data pipeline for the 2-D slice-based U-Net.

Key design decisions
--------------------
* Volumes are stored on disk at (182, 218, 182).  We treat the LAST axis as
  the slice axis (axial direction) and feed individual (182, 218) 2-D slices
  to the network.
* Each slice is centre-zero-padded from (182, 218) -> (256, 256) on-the-fly.
* The model accepts 3 channels [FLAIR, T1, T2] stacked along the last axis.
  Shape fed to the network: (256, 256, 3).
* Missing modalities (T1 or T2 absent for a study) are filled with zeros so
  that every study can still be used.
* Masks are thresholded at 0.5 so values are strictly {0, 1}.
* Blank slices (mask all-zero) are filtered at dataset construction time
  using SKIP_BLANK_RATIO.  Set to 0.0 to keep all slices.
"""

import os
import math
import numpy as np
import nibabel as nib
import tensorflow as tf

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_H = 256          # final spatial height fed to model
TARGET_W = 256          # final spatial width  fed to model
CHANNELS = 3            # FLAIR, T1w, T2w

DEFAULT_SKIP_BLANK_RATIO = 0.95  # used when caller doesn't pass a value


# ── Helpers ──────────────────────────────────────────────────────────────────

def centre_pad_2d(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Zero-pad a 2-D array to (th, tw), centred."""
    h, w = arr.shape
    pad_h = th - h
    pad_w = tw - w
    assert pad_h >= 0 and pad_w >= 0, \
        f"Source ({h},{w}) larger than target ({th},{tw})"
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left
    return np.pad(arr, ((top, bottom), (left, right)), mode="constant")


def load_volume(study_dir: str):
    """
    Load FLAIR, T1, T2 and MASK from study_dir.
    Returns (flair, t1, t2, mask) as float32 arrays of shape (H, W, D).
    T1 and T2 are optional; replaced with zeros if absent (graceful fallback).
    """
    def _load(fname):
        path = os.path.join(study_dir, fname)
        if not os.path.exists(path):
            return None
        data = nib.load(path).get_fdata(dtype=np.float32)
        # Proper MR Normalization: Min-Max scaling to [0, 1]
        dmin, dmax = data.min(), data.max()
        if dmax > dmin:
            data = (data - dmin) / (dmax - dmin)
        else:
            data = np.zeros_like(data)
        return data

    flair = _load("flair.nii.gz")
    mask_path = os.path.join(study_dir, "mask.nii.gz")
    mask = nib.load(mask_path).get_fdata(dtype=np.float32) if os.path.exists(mask_path) else None

    if flair is None or mask is None:
        raise FileNotFoundError(
            f"flair.nii.gz or mask.nii.gz missing in {study_dir}")

    shape = flair.shape                          # (H, W, D)
    t1 = _load("t1.nii.gz")
    t2 = _load("t2.nii.gz")
    if t1 is None:
        t1 = np.zeros(shape, dtype=np.float32)
    if t2 is None:
        t2 = np.zeros(shape, dtype=np.float32)

    return flair, t1, t2, (mask > 0.5).astype(np.float32)


def extract_slices(study_dir: str,
                   skip_blank_ratio: float = DEFAULT_SKIP_BLANK_RATIO,
                   is_train: bool = True):
    """
    Extract all 2-D axial slices from a study and return as numpy arrays.
    Each slice stacks [FLAIR, T1, T2] as a 3-channel image.

    Args:
        skip_blank_ratio : Fraction of lesion-free slices to discard during
                           training.  Range [0.0, 1.0].
                           0.0  = keep every slice (no filtering)
                           0.95 = drop 95% of background-only slices (default)
                           Set to 0.0 for validation / test splits.
    Returns:
        imgs  : (N, 256, 256, 3)  float32   channel order: FLAIR / T1 / T2
        masks : (N, 256, 256, 1)  float32
    """
    flair, t1, t2, mask = load_volume(study_dir)
    depth = flair.shape[2]

    imgs_list  = []
    masks_list = []

    rng = np.random.default_rng(seed=42)

    for z in range(depth):
        m_slice = mask[:, :, z]
        has_lesion = m_slice.max() > 0

        # Probabilistically skip blank slices during training
        if is_train and not has_lesion:
            if rng.random() < skip_blank_ratio:
                continue

        # Pad each modality and stack -> (256, 256, 3)   [FLAIR | T1 | T2]
        f_s  = centre_pad_2d(flair[:, :, z], TARGET_H, TARGET_W)
        t1_s = centre_pad_2d(t1[:, :, z],    TARGET_H, TARGET_W)
        t2_s = centre_pad_2d(t2[:, :, z],    TARGET_H, TARGET_W)

        img_slice  = np.stack([f_s, t1_s, t2_s], axis=-1)   # (256,256,3)
        msk_slice  = centre_pad_2d(m_slice, TARGET_H, TARGET_W)[..., np.newaxis]

        imgs_list.append(img_slice)
        masks_list.append(msk_slice)

    if not imgs_list:
        return None, None

    return (np.array(imgs_list, dtype=np.float32),
            np.array(masks_list, dtype=np.float32))


# ── Dataset builders ─────────────────────────────────────────────────────────

def _build_arrays(split_dir: str,
                  is_train: bool = True,
                  skip_blank_ratio: float = DEFAULT_SKIP_BLANK_RATIO):
    """Iterate all studies in split_dir and stack every slice."""
    all_imgs  = []
    all_masks = []

    studies = sorted(os.listdir(split_dir))
    for study_id in studies:
        study_path = os.path.join(split_dir, study_id)
        if not os.path.isdir(study_path):
            continue
        try:
            imgs, masks = extract_slices(study_path,
                                         skip_blank_ratio=skip_blank_ratio,
                                         is_train=is_train)
            if imgs is not None:
                all_imgs.append(imgs)
                all_masks.append(masks)
                print(f"  [{study_id}] {imgs.shape[0]} slices loaded")
        except Exception as e:
            print(f"  [WARN] {study_id}: {e}")

    if not all_imgs:
        return None, None

    X = np.concatenate(all_imgs,  axis=0)   # (total_slices, 256, 256, 3)
    Y = np.concatenate(all_masks, axis=0)   # (total_slices, 256, 256, 1)
    return X, Y


def build_tf_dataset(split_dir: str,
                     batch_size: int = 8,
                     is_train: bool = True,
                     skip_blank_ratio: float = DEFAULT_SKIP_BLANK_RATIO,
                     shuffle_buffer: int = 2000):
    """
    Build a tf.data.Dataset from a split directory.

    Args:
        skip_blank_ratio : Fraction of lesion-free slices to drop (training only).
                           Pass 0.0 to retain all slices (recommended for val/test).
    Returns: (tf.data.Dataset, n_slices)
    """
    print(f"\nBuilding dataset from: {split_dir}")
    X, Y = _build_arrays(split_dir, is_train=is_train,
                         skip_blank_ratio=skip_blank_ratio)

    if X is None:
        raise RuntimeError(f"No data found in {split_dir}")

    print(f"  Total slices: {X.shape[0]}  image shape: {X.shape[1:]}")

    with tf.device('/cpu:0'):
        ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if is_train:
        ds = ds.shuffle(buffer_size=min(shuffle_buffer, X.shape[0]))
    ds = (ds
          .batch(batch_size, drop_remainder=False)
          .prefetch(tf.data.AUTOTUNE))
    return ds, X.shape[0]
