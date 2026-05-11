"""
prepare_dataset.py
------------------
Creates a clean, model-ready copy of the preprocessed data in:
  /Volumes/Expansion1TB/MS/model_dataset/
  
Structure produced
------------------
model_dataset/
  train/
    <patient_timepoint>/
      flair.nii.gz
      t1.nii.gz
      t2.nii.gz
      mask.nii.gz
  test/                     (10% of MSLesSeg patients, held-out)
    ...
  external_val/             (ISBI2015 5 training subjects, zero-padded to 182x218x182)
    ...

Source directories pooled
-------------------------
* MSLesSeg Dataset/train/  -- 53 patients (P1-P53), timepoints in T1/T2/... subfolders
* MSLesSeg Dataset/test/   -- 22 patients (P54-P75), flat structure (no timepoint subfolders)
All 75 patients are pooled, then split 90/10 at the patient level.

Dimensions
----------
* MSLesSeg preprocessed:  (182, 218, 182)  -- used as-is
* ISBI2015 preprocessed:  (181, 217, 181)  -- zero-padded +1 on each axis -> (182, 218, 182)

The 2-D U-Net receives individual axial slices: (256, 256, 1) after
centre-padding x/y from 182x218 to 256x256.  That final padding is
done on-the-fly inside the DataLoader (dataset.py), NOT here.

Usage
-----
  python prepare_dataset.py
"""

import os
import shutil
import random
import numpy as np
import nibabel as nib

# ─────────────────────────────────────────────────────────────────────────────
# Paths – edit only these if you move data
# ─────────────────────────────────────────────────────────────────────────────
MSLESSEG_TRAIN = "/Volumes/Expansion1TB/MS/data/MSLesSeg/MSLesSeg Dataset/train"
MSLESSEG_TEST  = "/Volumes/Expansion1TB/MS/data/MSLesSeg/MSLesSeg Dataset/test"
ISBI_TRAIN     = "/Volumes/Expansion1TB/MS/data/ISBI2015/training"
OUT_ROOT       = "/Volumes/Expansion1TB/MS/model_dataset"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def copy_nii(src, dst):
    """Copy a NIfTI file, creating parent directories as needed."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  Copied  {os.path.basename(src):35s}  ->  {dst}")


def zero_pad_nii(src, dst, target_shape=(182, 218, 182)):
    """
    Load a NIfTI file, zero-pad it symmetrically to *target_shape* if
    needed, and save the result while preserving the affine header.
    """
    img  = nib.load(src)
    data = img.get_fdata(dtype=np.float32)
    src_shape = np.array(data.shape)
    tgt_shape = np.array(target_shape)

    if np.all(src_shape == tgt_shape):
        # Nothing to pad – just copy
        copy_nii(src, dst)
        return

    pad_total = tgt_shape - src_shape
    assert np.all(pad_total >= 0), \
        f"Source {src_shape} larger than target {tgt_shape}; cannot zero-pad."

    # Distribute padding: floor on left, ceil on right
    pad_before = pad_total // 2
    pad_after  = pad_total - pad_before
    pad_width  = [(pad_before[i], pad_after[i]) for i in range(3)]

    padded = np.pad(data, pad_width, mode="constant", constant_values=0)
    new_img = nib.Nifti1Image(padded, img.affine, img.header)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    nib.save(new_img, dst)
    print(f"  Padded  {os.path.basename(src):35s}  {tuple(src_shape)} -> {tuple(tgt_shape)}  ->  {dst}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MSLesSeg  (90 / 10 patient-level split across ALL 75 patients)
# ─────────────────────────────────────────────────────────────────────────────

def collect_mslesseg_studies(base_dir):
    """
    Collect studies from the MSLesSeg Dataset/train/ directory.
    Patients here have timepoint subfolders (T1, T2, T3 ...).
    Returns a list of (patient, timepoint_label, folder_path) tuples.
    """
    studies = []
    for patient in sorted(os.listdir(base_dir)):
        p_path = os.path.join(base_dir, patient)
        if not os.path.isdir(p_path) or not patient.startswith("P"):
            continue
        for tp in sorted(os.listdir(p_path)):
            tp_path = os.path.join(p_path, tp)
            if not os.path.isdir(tp_path) or not tp.startswith("T"):
                continue
            studies.append((patient, tp, tp_path))
    return studies


def collect_mslesseg_flat_studies(base_dir):
    """
    Collect studies from the MSLesSeg Dataset/test/ directory.
    Patients here have a flat structure — NIfTI files sit directly
    inside the patient folder (no T* subfolders).
    We label each such study as timepoint 'T1'.
    Returns a list of (patient, 'T1', folder_path) tuples.
    """
    studies = []
    for patient in sorted(os.listdir(base_dir)):
        p_path = os.path.join(base_dir, patient)
        if not os.path.isdir(p_path) or not patient.startswith("P"):
            continue
        # Confirm there are NIfTI files directly inside (flat layout)
        nii_files = [f for f in os.listdir(p_path)
                     if f.endswith(".nii") or f.endswith(".nii.gz")]
        if nii_files:
            studies.append((patient, "T1", p_path))
    return studies


def copy_mslesseg_study(patient, tp, tp_path, split_dir):
    """
    Find FLAIR / T1 / T2 / MASK inside tp_path and copy them into
    split_dir/<patient>_<tp>/.
    """
    study_id  = f"{patient}_{tp}"
    out_study = os.path.join(split_dir, study_id)
    os.makedirs(out_study, exist_ok=True)

    file_map = {}          # role -> source path
    for fname in os.listdir(tp_path):
        fu = fname.upper()
        if not (fname.endswith(".nii") or fname.endswith(".nii.gz")):
            continue
        if "MASK" in fu:
            file_map["mask"] = os.path.join(tp_path, fname)
        elif "FLAIR" in fu:
            file_map["flair"] = os.path.join(tp_path, fname)
        elif "T2" in fu:
            file_map["t2"] = os.path.join(tp_path, fname)
        elif "T1" in fu:
            file_map["t1"] = os.path.join(tp_path, fname)

    required = ["flair", "mask"]
    missing  = [r for r in required if r not in file_map]
    if missing:
        print(f"  [WARN] {study_id}: missing {missing}, skipping.")
        return False

    for role, src in file_map.items():
        dst = os.path.join(out_study, f"{role}.nii.gz")
        copy_nii(src, dst)
    return True


def prepare_mslesseg(): # 90/10 Patient-level split (train/test) — all 75 patients
    print("\n" + "="*60)
    print("MSLesSeg: collecting ALL 75 patients (train + test sources) ...")
    print("="*60)

    # Pool studies from both MSLesSeg source directories
    studies_train_src = collect_mslesseg_studies(MSLESSEG_TRAIN)      # 53 patients, T* subfolders
    studies_test_src  = collect_mslesseg_flat_studies(MSLESSEG_TEST)  # 22 patients, flat layout
    all_studies = studies_train_src + studies_test_src

    print(f"  MSLesSeg Dataset/train source : {len(studies_train_src)} studies "
          f"({len(set(s[0] for s in studies_train_src))} patients)")
    print(f"  MSLesSeg Dataset/test  source : {len(studies_test_src)} studies "
          f"({len(set(s[0] for s in studies_test_src))} patients)")
    print(f"  Total pooled                  : {len(all_studies)} studies "
          f"({len(set(s[0] for s in all_studies))} patients)")

    # ── Patient-level split ───────────────────────────────────────────────────
    from collections import defaultdict
    patient_to_studies = defaultdict(list)
    for patient, tp, tp_path in all_studies:
        patient_to_studies[patient].append((patient, tp, tp_path))

    patients = sorted(patient_to_studies.keys())
    random.shuffle(patients)                        # reproducible via SEED=42
    split_idx      = int(len(patients) * 0.9)
    train_patients = patients[:split_idx]
    test_patients  = patients[split_idx:]

    train_studies = [s for p in train_patients for s in patient_to_studies[p]]
    test_studies  = [s for p in test_patients  for s in patient_to_studies[p]]

    print(f"\n  -> 90% train : {len(train_patients)} patients  ({len(train_studies)} studies)")
    print(f"  -> 10% test  : {len(test_patients)} patients  ({len(test_studies)} studies)")
    print(f"  Train patients: {train_patients}")
    print(f"  Test  patients: {test_patients}")

    for patient, tp, tp_path in train_studies:
        print(f"\n[TRAIN] {patient}/{tp}")
        copy_mslesseg_study(patient, tp, tp_path,
                            os.path.join(OUT_ROOT, "train"))

    for patient, tp, tp_path in test_studies:
        print(f"\n[TEST] {patient}/{tp}")
        copy_mslesseg_study(patient, tp, tp_path,
                            os.path.join(OUT_ROOT, "test"))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ISBI2015 external validation (5 subjects, zero-padded)
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: ISBI provides TWO rater masks per timepoint (mask1, mask2).
#       We use mask1 (Rater-1) as the ground truth for external validation.
#       Change MASK_RATER below to "mask2" to use Rater-2 instead.
MASK_RATER = "mask1"

def prepare_isbi():
    print("\n" + "="*60)
    print("ISBI2015: collecting studies for external validation ...")
    print("="*60)

    ext_dir = os.path.join(OUT_ROOT, "external_val")
    subjects = sorted([
        d for d in os.listdir(ISBI_TRAIN)
        if d.startswith("training") and
        os.path.isdir(os.path.join(ISBI_TRAIN, d))
    ])
    print(f"Total ISBI subjects: {len(subjects)}")

    for subj in subjects:
        subj_path = os.path.join(ISBI_TRAIN, subj)
        pp_path   = os.path.join(subj_path, "preprocessed")
        msk_path  = os.path.join(subj_path, "masks")

        if not os.path.isdir(pp_path):
            print(f"  [WARN] {subj}: no preprocessed folder, skipping.")
            continue

        # Discover timepoints from preprocessed filenames
        timepoints = sorted(set(
            f.split("_")[1]
            for f in os.listdir(pp_path)
            if f.endswith("_pp.nii")
        ))

        for tp in timepoints:
            study_id  = f"{subj}_{tp}"
            out_study = os.path.join(ext_dir, study_id)
            print(f"\n[EXT_VAL] {study_id}")

            def pp_file(modality):
                """Return path to preprocessed file for given modality."""
                fname = f"{subj}_{tp}_{modality}_pp.nii"
                return os.path.join(pp_path, fname)

            def msk_file(rater):
                fname = f"{subj}_{tp}_{rater}.nii"
                return os.path.join(msk_path, fname)

            file_map = {
                "flair": pp_file("flair"),
                "t1":    pp_file("mprage"),
                "t2":    pp_file("t2"),
                "pd":    pp_file("pd"),
                "mask":  msk_file(MASK_RATER),
            }

            for role, src in file_map.items():
                if not os.path.exists(src):
                    print(f"  [WARN] {study_id}: {role} not found at {src}, skipping file.")
                    continue
                dst = os.path.join(out_study, f"{role}.nii.gz")
                zero_pad_nii(src, dst, target_shape=(182, 218, 182))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    prepare_mslesseg()
    prepare_isbi()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Dataset preparation complete.")
    print("="*60)
    for split in ["train", "test", "external_val"]:
        split_path = os.path.join(OUT_ROOT, split)
        n = len(os.listdir(split_path)) if os.path.isdir(split_path) else 0
        print(f"  {split:15s}: {n} studies")
    print(f"\nDataset root: {OUT_ROOT}")
