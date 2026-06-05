import os
import random
import numpy as np
import nibabel as nib
import tensorflow as tf

def centre_pad_crop_3d(arr: np.ndarray, target_h: int = 256, target_w: int = 256, target_d: int = 192) -> np.ndarray:
    """Center crop or symmetrically zero-pad a 3-D array to exactly (target_h, target_w, target_d)."""
    h, w, d = arr.shape
    
    # 1. Crop if larger
    if h > target_h:
        sh = (h - target_h) // 2
        arr = arr[sh:sh + target_h, :, :]
        h = target_h
    if w > target_w:
        sw = (w - target_w) // 2
        arr = arr[:, sw:sw + target_w, :]
        w = target_w
    if d > target_d:
        sd = (d - target_d) // 2
        arr = arr[:, :, sd:sd + target_d]
        d = target_d
        
    # 2. Pad if smaller
    pad_h = target_h - h
    pad_w = target_w - w
    pad_d = target_d - d
    
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        front = pad_d // 2
        back = pad_d - front
        arr = np.pad(arr, ((top, bottom), (left, right), (front, back)), mode="constant", constant_values=0)
        
    return arr

def load_and_preprocess_subject_3d(subj_dir: str, target_h: int = 256, target_w: int = 256, target_d: int = 192):
    """
    Loads T1, T2, FLAIR and Mask volumes for a subject.
    Normalizes each modality individually.
    Pads/crops them on-the-fly to exactly (target_h, target_w, target_d).
    Stacks modalities along the channel dimension.
    """
    t1_path = os.path.join(subj_dir, "t1.nii.gz")
    t2_path = os.path.join(subj_dir, "t2.nii.gz")
    flair_path = os.path.join(subj_dir, "flair.nii.gz")
    mask_path = os.path.join(subj_dir, "mask.nii.gz")

    if not (os.path.exists(t1_path) and os.path.exists(t2_path) and os.path.exists(flair_path) and os.path.exists(mask_path)):
        return None, None

    t1_data = nib.load(t1_path).get_fdata(dtype=np.float32)
    t2_data = nib.load(t2_path).get_fdata(dtype=np.float32)
    flair_data = nib.load(flair_path).get_fdata(dtype=np.float32)
    mask_data = (nib.load(mask_path).get_fdata(dtype=np.float32) > 0.5).astype(np.float32)

    def normalize(arr):
        dmin, dmax = arr.min(), arr.max()
        if dmax > dmin:
            return (arr - dmin) / (dmax - dmin)
        return np.zeros_like(arr)

    # 1. Normalize
    t1_data = normalize(t1_data)
    t2_data = normalize(t2_data)
    flair_data = normalize(flair_data)

    # 2. Symmetrically pad/crop to target shape
    t1_prep = centre_pad_crop_3d(t1_data, target_h, target_w, target_d)
    t2_prep = centre_pad_crop_3d(t2_data, target_h, target_w, target_d)
    flair_prep = centre_pad_crop_3d(flair_data, target_h, target_w, target_d)
    mask_prep = centre_pad_crop_3d(mask_data, target_h, target_w, target_d)

    # 3. Stack channels -> (H, W, D, 3)
    img_3d = np.stack([t1_prep, t2_prep, flair_prep], axis=-1)
    mask_3d = mask_prep[..., np.newaxis] # (H, W, D, 1)

    return img_3d, mask_3d

def get_splits(preprocessed_dir: str, split_seed: int = 42):
    """Scans folder and splits subjects into deterministic lists."""
    subjects = sorted([d for d in os.listdir(preprocessed_dir) if d.startswith("S") and os.path.isdir(os.path.join(preprocessed_dir, d))])
    
    rng = random.Random(split_seed)
    shuffled_subjects = list(subjects)
    rng.shuffle(shuffled_subjects)
    
    n_subjects = len(shuffled_subjects)
    n_train = int(0.8 * n_subjects)
    n_val = int(0.1 * n_subjects)
    
    train_subjs = shuffled_subjects[:n_train]
    val_subjs = shuffled_subjects[n_train:n_train + n_val]
    test_subjs = shuffled_subjects[n_train + n_val:]
    
    return train_subjs, val_subjs, test_subjs

def make_generator(subjects_list, preprocessed_dir, target_h, target_w, target_d, shuffle=False):
    """Creates a generator that loads and yields one 3D volume at a time."""
    def generator():
        subjs = list(subjects_list)
        if shuffle:
            random.shuffle(subjs)
            
        for subject_id in subjs:
            subj_dir = os.path.join(preprocessed_dir, subject_id)
            img, mask = load_and_preprocess_subject_3d(subj_dir, target_h, target_w, target_d)
            if img is not None:
                yield img, mask
                
    return generator

def build_datasets(preprocessed_dir: str,
                   batch_size: int = 1,
                   target_h: int = 256,
                   target_w: int = 256,
                   target_d: int = 192,
                   split_seed: int = 42):
    """
    Builds data loader tf.data.Dataset objects for 3-D volumes using generators
    to prevent memory exhaustion.
    """
    train_subjs, val_subjs, test_subjs = get_splits(preprocessed_dir, split_seed)
    
    print(f"Splitting {len(train_subjs)+len(val_subjs)+len(test_subjs)} subjects for 3D UNet:")
    print(f"  Train : {len(train_subjs)} subjects")
    print(f"  Val   : {len(val_subjs)} subjects")
    print(f"  Test  : {len(test_subjs)} subjects")
    
    output_signature = (
        tf.TensorSpec(shape=(target_h, target_w, target_d, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(target_h, target_w, target_d, 1), dtype=tf.float32)
    )
    
    train_gen = make_generator(train_subjs, preprocessed_dir, target_h, target_w, target_d, shuffle=True)
    val_gen = make_generator(val_subjs, preprocessed_dir, target_h, target_w, target_d, shuffle=False)
    test_gen = make_generator(test_subjs, preprocessed_dir, target_h, target_w, target_d, shuffle=False)
    
    # Build datasets from generators
    train_ds = tf.data.Dataset.from_generator(train_gen, output_signature=output_signature)
    val_ds = tf.data.Dataset.from_generator(val_gen, output_signature=output_signature)
    test_ds = tf.data.Dataset.from_generator(test_gen, output_signature=output_signature)
    
    # Batch and prefetch (batch_size is usually 1 for 3D UNet to avoid VRAM OOM)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds, test_ds, {
        "train": train_subjs,
        "val": val_subjs,
        "test": test_subjs
    }
