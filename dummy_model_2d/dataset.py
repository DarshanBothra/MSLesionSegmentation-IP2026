"""
dataset.py
----------
TensorFlow/Keras data pipeline for the 2-D slice-based U-Net.

Key design decisions
--------------------
* Dataset is a flat directory of SXXX_FLAIR.nii.gz + SXXX_MASK.nii.gz pairs.
* Volumes are treated slice-by-slice along the LAST (axial) axis.
* Each slice is centre-zero-padded to (256, 256) on-the-fly.
* The model accepts 1 channel [FLAIR only]. Shape fed to network: (256, 256, 1).
* Intensities are Min-Max normalised to [0, 1] per volume.
* Masks are thresholded at 0.5 so values are strictly {0, 1}.
* Blank slices (mask all-zero) are filtered during training using SKIP_BLANK_RATIO.
  Set to 0.0 to keep all slices (recommended for val/test).
"""

import os
import numpy as np
import nibabel as nib
import tensorflow as tf

# ── Constants ────────────────────────────────────────────────────────────────
TARGET_H = 256          # final spatial height fed to model
TARGET_W = 256          # final spatial width  fed to model
CHANNELS = 1            # FLAIR only

DEFAULT_SKIP_BLANK_RATIO = 0.95  # used when caller doesn't pass a value


# ── Helpers ──────────────────────────────────────────────────────────────────

def centre_pad_2d(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Zero-pad a 2-D array to (th, tw), centred."""
    h, w = arr.shape
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left
    return np.pad(arr, ((top, bottom), (left, right)), mode="constant")


def load_volume(flair_path: str, mask_path: str):
    """
    Load FLAIR and MASK from explicit file paths.
    Returns (flair, mask) as float32 arrays of shape (H, W, D).
    FLAIR is Min-Max normalised to [0, 1].
    Mask is thresholded to binary {0, 1}.
    """
    flair_data = nib.load(flair_path).get_fdata(dtype=np.float32)
    mask_data  = nib.load(mask_path).get_fdata(dtype=np.float32)

    # Min-Max normalisation per volume
    dmin, dmax = flair_data.min(), flair_data.max()
    if dmax > dmin:
        flair_data = (flair_data - dmin) / (dmax - dmin)
    else:
        flair_data = np.zeros_like(flair_data)

    mask_data = (mask_data > 0.5).astype(np.float32)
    return flair_data, mask_data


def extract_slices(flair_path: str,
                   mask_path: str,
                   skip_blank_ratio: float = DEFAULT_SKIP_BLANK_RATIO,
                   is_train: bool = True):
    """
    Extract 2-D axial slices from a FLAIR/MASK pair.

    Returns:
        imgs  : (N, 256, 256, 1)  float32   — FLAIR channel
        masks : (N, 256, 256, 1)  float32
    """
    flair, mask = load_volume(flair_path, mask_path)
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

        f_s = centre_pad_2d(flair[:, :, z], TARGET_H, TARGET_W)

        img_slice = f_s[..., np.newaxis]                             # (256, 256, 1)
        msk_slice = centre_pad_2d(m_slice, TARGET_H, TARGET_W)[..., np.newaxis]

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
    """
    Scan split_dir for SXXX_FLAIR.nii.gz / SXXX_MASK.nii.gz pairs
    and concatenate all slices into numpy arrays.
    """
    all_imgs  = []
    all_masks = []

    # Collect all FLAIR files and match with their MASK
    flair_files = sorted(
        f for f in os.listdir(split_dir) if f.endswith("_FLAIR.nii.gz")
    )

    if not flair_files:
        raise RuntimeError(f"No *_FLAIR.nii.gz files found in {split_dir}")

    for flair_fname in flair_files:
        subject_id = flair_fname.replace("_FLAIR.nii.gz", "")
        mask_fname = f"{subject_id}_MASK.nii.gz"

        flair_path = os.path.join(split_dir, flair_fname)
        mask_path  = os.path.join(split_dir, mask_fname)

        if not os.path.exists(mask_path):
            print(f"  [WARN] No mask for {subject_id} — skipping")
            continue

        try:
            imgs, masks = extract_slices(flair_path, mask_path,
                                         skip_blank_ratio=skip_blank_ratio,
                                         is_train=is_train)
            if imgs is not None:
                all_imgs.append(imgs)
                all_masks.append(masks)
                print(f"  [{subject_id}] {imgs.shape[0]} slices loaded")
        except Exception as e:
            print(f"  [WARN] {subject_id}: {e}")

    if not all_imgs:
        return None, None

    X = np.concatenate(all_imgs,  axis=0)   # (total_slices, 256, 256, 1)
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
        split_dir        : Path to train / val / test split folder.
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
