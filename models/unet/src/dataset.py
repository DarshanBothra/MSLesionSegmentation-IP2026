"""
dataset.py
----------
TensorFlow/Keras data pipeline for the 2-D axial slice-based multi-modal U-Net.
"""

import os
import random
import numpy as np
import nibabel as nib
import tensorflow as tf

def centre_pad_crop_2d(arr: np.ndarray, target_h: int = 182, target_w: int = 218) -> np.ndarray:
    """Center crop or symmetrically zero-pad a 2-D array to exactly (target_h, target_w)."""
    h, w = arr.shape
    
    # 1. Crop if larger
    if h > target_h:
        sh = (h - target_h) // 2
        arr = arr[sh:sh + target_h, :]
        h = target_h
    if w > target_w:
        sw = (w - target_w) // 2
        arr = arr[:, sw:sw + target_w]
        w = target_w
        
    # 2. Pad if smaller
    pad_h = target_h - h
    pad_w = target_w - w
    
    if pad_h > 0 or pad_w > 0:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        arr = np.pad(arr, ((top, bottom), (left, right)), mode="constant", constant_values=0)
        
    return arr

def load_volume(t1_path: str, t2_path: str, flair_path: str, mask_path: str):
    """
    Load T1, T2, FLAIR and MASK from explicit file paths.
    Returns (t1, t2, flair, mask) as float32 arrays.
    Each modality is Min-Max normalised to [0, 1] individually per volume.
    Mask is thresholded to binary {0, 1}.
    """
    t1_data = nib.load(t1_path).get_fdata(dtype=np.float32)
    t2_data = nib.load(t2_path).get_fdata(dtype=np.float32)
    flair_data = nib.load(flair_path).get_fdata(dtype=np.float32)
    mask_data = nib.load(mask_path).get_fdata(dtype=np.float32)

    def normalize(arr):
        dmin, dmax = arr.min(), arr.max()
        if dmax > dmin:
            return (arr - dmin) / (dmax - dmin)
        return np.zeros_like(arr)

    t1_data = normalize(t1_data)
    t2_data = normalize(t2_data)
    flair_data = normalize(flair_data)
    mask_data = (mask_data > 0.5).astype(np.float32)

    return t1_data, t2_data, flair_data, mask_data

def extract_slices(t1_path: str,
                   t2_path: str,
                   flair_path: str,
                   mask_path: str,
                   target_h: int = 182,
                   target_w: int = 218,
                   skip_blank_ratio: float = 0.95,
                   is_train: bool = True):
    """
    Extract 2-D axial slices from a multi-modal set of volumes.
    Stacks T1, T2, and FLAIR to create a 3-channel slice.
    
    Returns:
        imgs  : (N, target_h, target_w, 3)  float32
        masks : (N, target_h, target_w, 1)  float32
    """
    t1, t2, flair, mask = load_volume(t1_path, t2_path, flair_path, mask_path)
    depth = flair.shape[2]
    
    imgs_list = []
    masks_list = []
    
    rng = np.random.default_rng(seed=42)
    
    for z in range(depth):
        m_slice = mask[:, :, z] if z < mask.shape[2] else np.zeros(mask.shape[:2], dtype=np.float32)
        has_lesion = m_slice.max() > 0
        
        # Probabilistically skip blank slices during training
        if is_train and not has_lesion:
            if rng.random() < skip_blank_ratio:
                continue
                
        # Pad/crop each modality slice
        t1_s = centre_pad_crop_2d(t1[:, :, z], target_h, target_w)
        t2_s = centre_pad_crop_2d(t2[:, :, z], target_h, target_w)
        fl_s = centre_pad_crop_2d(flair[:, :, z], target_h, target_w)
        
        # Stack channels
        img_slice = np.stack([t1_s, t2_s, fl_s], axis=-1)  # (target_h, target_w, 3)
        msk_slice = centre_pad_crop_2d(m_slice, target_h, target_w)[..., np.newaxis] # (target_h, target_w, 1)
        
        imgs_list.append(img_slice)
        masks_list.append(msk_slice)
        
    if not imgs_list:
        return None, None
        
    return (np.array(imgs_list, dtype=np.float32),
            np.array(masks_list, dtype=np.float32))

def _build_arrays(subjects_list: list,
                  preprocessed_dir: str,
                  is_train: bool = True,
                  skip_blank_ratio: float = 0.95,
                  target_h: int = 182,
                  target_w: int = 218):
    """Load slices for a list of subjects and concatenate into numpy arrays."""
    all_imgs = []
    all_masks = []
    
    for subject_id in subjects_list:
        subj_dir = os.path.join(preprocessed_dir, subject_id)
        t1_path = os.path.join(subj_dir, "t1.nii.gz")
        t2_path = os.path.join(subj_dir, "t2.nii.gz")
        flair_path = os.path.join(subj_dir, "flair.nii.gz")
        mask_path = os.path.join(subj_dir, "mask.nii.gz")
        
        if not (os.path.exists(t1_path) and os.path.exists(t2_path) and os.path.exists(flair_path) and os.path.exists(mask_path)):
            print(f"  [WARN] Subject {subject_id} is missing one or more files — skipping")
            continue
            
        try:
            imgs, masks = extract_slices(t1_path, t2_path, flair_path, mask_path,
                                         target_h=target_h, target_w=target_w,
                                         skip_blank_ratio=skip_blank_ratio,
                                         is_train=is_train)
            if imgs is not None:
                all_imgs.append(imgs)
                all_masks.append(masks)
                print(f"  [{subject_id}] {imgs.shape[0]} slices loaded")
        except Exception as e:
            print(f"  [WARN] Subject {subject_id} failed: {e}")
            
    if not all_imgs:
        return None, None
        
    X = np.concatenate(all_imgs, axis=0)
    Y = np.concatenate(all_masks, axis=0)
    return X, Y

def build_datasets(preprocessed_dir: str,
                   batch_size: int = 4,
                   skip_blank_ratio: float = 0.95,
                   target_h: int = 182,
                   target_w: int = 218,
                   split_seed: int = 42):
    """
    Scan preprocessed_dir, split subjects deterministically,
    load training, validation, and testing slices, and return tf.data.Datasets.
    """
    # Scan all subject folders (starting with 'S')
    subjects = sorted([d for d in os.listdir(preprocessed_dir) if d.startswith("S") and os.path.isdir(os.path.join(preprocessed_dir, d))])
    
    # Deterministic split (80% Train, 10% Val, 10% Test)
    rng = random.Random(split_seed)
    shuffled_subjects = list(subjects)
    rng.shuffle(shuffled_subjects)
    
    n_subjects = len(shuffled_subjects)
    n_train = int(0.8 * n_subjects)
    n_val = int(0.1 * n_subjects)
    
    train_subjs = shuffled_subjects[:n_train]
    val_subjs = shuffled_subjects[n_train:n_train + n_val]
    test_subjs = shuffled_subjects[n_train + n_val:]
    
    print(f"Splitting {n_subjects} subjects:")
    print(f"  Train : {len(train_subjs)} subjects")
    print(f"  Val   : {len(val_subjs)} subjects")
    print(f"  Test  : {len(test_subjs)} subjects")
    
    print("\nLoading Train Split...")
    X_train, Y_train = _build_arrays(train_subjs, preprocessed_dir, is_train=True,
                                     skip_blank_ratio=skip_blank_ratio, target_h=target_h, target_w=target_w)
    
    print("\nLoading Val Split...")
    X_val, Y_val = _build_arrays(val_subjs, preprocessed_dir, is_train=False,
                                 skip_blank_ratio=0.0, target_h=target_h, target_w=target_w)
                                 
    print("\nLoading Test Split...")
    X_test, Y_test = _build_arrays(test_subjs, preprocessed_dir, is_train=False,
                                   skip_blank_ratio=0.0, target_h=target_h, target_w=target_w)
                                   
    if X_train is None or X_val is None or X_test is None:
        raise RuntimeError("Failed to build one or more dataset splits.")
        
    print(f"\nFinal dataset slice counts:")
    print(f"  Train : {X_train.shape[0]} slices  (images: {X_train.shape[1:]}, masks: {Y_train.shape[1:]})")
    print(f"  Val   : {X_val.shape[0]} slices  (images: {X_val.shape[1:]}, masks: {Y_val.shape[1:]})")
    print(f"  Test  : {X_test.shape[0]} slices  (images: {X_test.shape[1:]}, masks: {Y_test.shape[1:]})")
    
    with tf.device('/cpu:0'):
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        
    train_ds = train_ds.shuffle(buffer_size=min(2000, X_train.shape[0])).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds, {
        "train": (X_train, Y_train, train_subjs),
        "val": (X_val, Y_val, val_subjs),
        "test": (X_test, Y_test, test_subjs)
    }
