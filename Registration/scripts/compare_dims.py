import os
import nibabel as nib

index = {
    "ISBI": {
        "FLAIR": "/Users/darshan/Code/Registration/data/ISBI/orig/training04_01_flair.nii.gz",
        "T1": "/Users/darshan/Code/Registration/data/ISBI/orig/training04_01_mprage.nii.gz",
        "T2": "/Users/darshan/Code/Registration/data/ISBI/orig/training04_01_t2.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/data/ISBI/masks/training04_01_mask1.nii"
    },
    "MICCAI": {
        "FLAIR": "/Users/darshan/Code/Registration/data/MICCAI/Raw_Data/FLAIR.nii.gz",
        "T1": "/Users/darshan/Code/Registration/data/MICCAI/Raw_Data/T1.nii.gz",
        "T2": "/Users/darshan/Code/Registration/data/MICCAI/Raw_Data/T2.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/data/MICCAI/Masks/Consensus.nii.gz",
    },
    "MSLegSeg": {
        "FLAIR": "/Users/darshan/Code/Registration/data/MSLegSeg/Raw/P6_T2_FLAIR.nii.gz",
        "T1": "/Users/darshan/Code/Registration/data/MSLegSeg/Raw/P6_T2_T1.nii.gz",
        "T2": "/Users/darshan/Code/Registration/data/MSLegSeg/Raw/P6_T2_T2.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/data/MSLegSeg/Raw/P6_T2_MASK.nii.gz",
    }
}
index_preprocessed = {
    "ISBI": {
        "FLAIR": "/Users/darshan/Code/Registration/data/ISBI/preprocessed/training04_01_flair_pp.nii",
        "T1": "/Users/darshan/Code/Registration/data/ISBI/preprocessed/training04_01_mprage_pp.nii",
        "T2": "/Users/darshan/Code/Registration/data/ISBI/preprocessed/training04_01_t2_pp.nii",
        "MASK": "/Users/darshan/Code/Registration/data/ISBI/masks/training04_01_mask1.nii"
    },
    "MICCAI": {
        "FLAIR": "/Users/darshan/Code/Registration/data/MICCAI/Preprocessed_Data/FLAIR_preprocessed.nii.gz",
        "T1": "/Users/darshan/Code/Registration/data/MICCAI/Preprocessed_Data/T1_preprocessed.nii.gz",
        "T2": "/Users/darshan/Code/Registration/data/MICCAI/Preprocessed_Data/T2_preprocessed.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/data/MICCAI/Masks/Consensus.nii.gz",
    },
    "MSLegSeg": {
        "FLAIR": "/Users/darshan/Code/Registration/data/MSLegSeg/Preprocessed/P6_T2_FLAIR.nii.gz",
        "T1": "/Users/darshan/Code/Registration/data/MSLegSeg/Preprocessed/P6_T2_T1.nii.gz",
        "T2": "/Users/darshan/Code/Registration/data/MSLegSeg/Preprocessed/P6_T2_T2.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/data/MSLegSeg/Raw/P6_T2_MASK.nii.gz",
    }
}

MNI_TEMPLATE = "/Users/darshan/Code/Registration/data/MNI152_T1_1mm.nii.gz"
for dataset in index:
    print(f"========== {dataset} ==========\n")
    flair_raw_path = index[dataset]["FLAIR"]
    flair_preprocessed_path = index_preprocessed[dataset]["FLAIR"]

    t1_raw_path = index[dataset]["T1"]
    t1_preprocessed_path = index_preprocessed[dataset]["T1"]

    t2_raw_path = index[dataset]["T2"]
    t2_preprocessed_path = index_preprocessed[dataset]["T2"]

    mask_raw_path = index[dataset]["MASK"]
    mask_preprocessed_path = index_preprocessed[dataset]["MASK"]

    flair_raw_orientation = nib.orientations.aff2axcodes(nib.load(flair_raw_path).affine)
    flair_preprocessed_orientation = nib.orientations.aff2axcodes(nib.load(flair_preprocessed_path).affine)

    print("FLAIR Raw Image orientation: ", flair_raw_orientation)
    print("FLAIR Preprocessed Image orientation: ", flair_preprocessed_orientation)
    print()

    t1_raw_orientation = nib.orientations.aff2axcodes(nib.load(t1_raw_path).affine)
    t1_preprocessed_orientation = nib.orientations.aff2axcodes(nib.load(t1_preprocessed_path).affine)

    print("T1 Raw Image orientation: ", t1_raw_orientation)
    print("T1 Preprocessed Image orientation: ", t1_preprocessed_orientation)
    print()

    t2_raw_orientation = nib.orientations.aff2axcodes(nib.load(t2_raw_path).affine)
    t2_preprocessed_orientation = nib.orientations.aff2axcodes(nib.load(t2_preprocessed_path).affine)

    print("T2 Raw Image orientation: ", t2_raw_orientation)
    print("T2 Preprocessed Image orientation: ", t2_preprocessed_orientation)
    print()

    mask_raw_orientation = nib.orientations.aff2axcodes(nib.load(mask_raw_path).affine)
    mask_preprocessed_orientation = nib.orientations.aff2axcodes(nib.load(mask_preprocessed_path).affine)

    print("MASK Raw Image orientation: ", mask_raw_orientation)
    print("MASK Preprocessed Image orientation: ", mask_preprocessed_orientation)
    print()

    print("==============================\n")

print("========== TEMPLATE ==========\n")
print(nib.orientations.aff2axcodes(nib.load(MNI_TEMPLATE).affine))
print("==============================")