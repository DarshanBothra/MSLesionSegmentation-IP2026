import os
import nibabel as nib
import numpy as np
import json

DATASET_ROOT = "/home/darshan/MS/data/PREPROCESSED/"
OUTPUT_JSON_PATH = "/home/darshan/MS/eda/results/preprocessing/samples.json"

def analyse_subject(root, subject):
    PATIENT_DIR = os.path.join(root, subject)
    subject_info = {}

    flair_path = os.path.join(PATIENT_DIR, "flair.nii.gz")
    t1_path = os.path.join(PATIENT_DIR, "t1.nii.gz")
    t2_path = os.path.join(PATIENT_DIR, "t2.nii.gz")
    mask_path = os.path.join(PATIENT_DIR, "mask.nii.gz")

    subject_info["flair_path"] = flair_path if os.path.exists(flair_path) else None
    subject_info["t1_path"] = t1_path if os.path.exists(t1_path) else None
    subject_info["t2_path"] = t2_path if os.path.exists(t2_path) else None
    subject_info["mask_path"] = mask_path if os.path.exists(mask_path) else None

    subject_info["flair_present"] = True if subject_info["flair_path"] else False
    subject_info["t1_present"] = True if subject_info["t1_path"] else False
    subject_info["t2_present"] = True if subject_info["t2_path"] else False
    subject_info["mask_present"] = True if subject_info["mask_path"] else False

    # dimensions

    flair_img = nib.load(subject_info["flair_path"]) if subject_info["flair_path"] else None
    t1_img = nib.load(subject_info["t1_path"]) if subject_info["t1_path"] else None
    t2_img = nib.load(subject_info["t2_path"]) if subject_info["t2_path"] else None
    mask_img = nib.load(subject_info["mask_path"]) if subject_info["mask_path"] else None

    flair_dims = flair_img.get_fdata().shape if subject_info["flair_path"] else None 
    t1_dims = t1_img.get_fdata().shape if subject_info["t1_path"] else None
    t2_dims = t2_img.get_fdata().shape if subject_info["t2_path"] else None
    mask_dims = mask_img.get_fdata().shape if subject_info["mask_path"] else None

    flair_orientation = nib.orientations.aff2axcodes(flair_img.affine) if subject_info["flair_path"] else None
    t1_orientation = nib.orientations.aff2axcodes(t1_img.affine) if subject_info["t1_path"] else None
    t2_orientation = nib.orientations.aff2axcodes(t2_img.affine) if subject_info["t2_path"] else None
    mask_orientation = nib.orientations.aff2axcodes(mask_img.affine) if subject_info["mask_path"] else None

    flair_voxel_size = flair_img.header.get_zooms() if subject_info["flair_path"] else None
    t1_voxel_size = t1_img.header.get_zooms() if subject_info["t1_path"] else None
    t2_voxel_size = t2_img.header.get_zooms() if subject_info["t2_path"] else None
    mask_voxel_size = mask_img.header.get_zooms() if subject_info["mask_path"] else None

    subject_info["flair_dims"] = str(flair_dims)
    subject_info["t1_dims"] = str(t1_dims)
    subject_info["t2_dims"] = str(t2_dims)
    subject_info["mask_dims"] = str(mask_dims)

    subject_info["flair_orientation"] = str(flair_orientation)
    subject_info["t1_orientation"] = str(t1_orientation)
    subject_info["t2_orientation"] = str(t2_orientation)
    subject_info["mask_orientation"] = str(mask_orientation)

    subject_info["flair_voxel_size"] = str(flair_voxel_size)
    subject_info["t1_voxel_size"] = str(t1_voxel_size)
    subject_info["t2_voxel_size"] = str(t2_voxel_size)
    subject_info["mask_voxel_size"] = str(mask_voxel_size)

    return subject_info

def main():
    output = {}
    subjects = sorted([x for x in os.listdir(DATASET_ROOT)])
    output["total_subjects"] = len(subjects)

    output["all_flair_present"] = True
    output["all_t1_present"] = True
    output["all_t2_present"] = True
    output["all_mask_present"] = True

    output["total_flair"] = 0
    output["total_t1"] = 0
    output["total_t2"] = 0
    output["total_mask"] = 0

    dims = {}
    voxels = {}
    orientations = {}

    flair_dims = {}
    t1_dims = {}
    t2_dims = {}
    mask_dims = {}

    flair_voxels = {}
    t1_voxels = {}
    t2_voxels = {}
    mask_voxels = {}
    
    flair_orientation = {}
    t1_orientation = {}
    t2_orientation = {}
    mask_orientation = {}

    for subject in subjects:
        subject_info = analyse_subject(DATASET_ROOT, subject)
        output[subject] = subject_info
        output["total_flair"] += int(subject_info["flair_present"])
        output["total_t1"] += int(subject_info["t1_present"])
        output["total_t2"] += int(subject_info["t2_present"])
        output["total_mask"] += int(subject_info["mask_present"])

        flair_dims[str(subject_info["flair_dims"])] = flair_dims.get(str(subject_info["flair_dims"]), 0) + 1
        t1_dims[str(subject_info["t1_dims"])] = t1_dims.get(str(subject_info["t1_dims"]), 0) + 1        
        t2_dims[str(subject_info["t2_dims"])] = t2_dims.get(str(subject_info["t2_dims"]), 0) + 1        
        mask_dims[str(subject_info["mask_dims"])] = mask_dims.get(str(subject_info["mask_dims"]), 0) + 1     

        flair_orientation[str(subject_info["flair_orientation"])] = flair_orientation.get(str(subject_info["flair_orientation"]), 0) + 1
        t1_orientation[str(subject_info["t1_orientation"])] = t1_orientation.get(str(subject_info["t1_orientation"]), 0) + 1
        t2_orientation[str(subject_info["t2_orientation"])] = t2_orientation.get(str(subject_info["t2_orientation"]), 0) + 1
        mask_orientation[str(subject_info["mask_orientation"])] = mask_orientation.get(str(subject_info["mask_orientation"]), 0) + 1

        flair_voxels[str(subject_info["flair_voxel_size"])] = flair_voxels.get(str(subject_info["flair_voxel_size"]), 0) + 1
        t1_voxels[str(subject_info["t1_voxel_size"])] = t1_voxels.get(str(subject_info["t1_voxel_size"]), 0) + 1
        t2_voxels[str(subject_info["t2_voxel_size"])] = t2_voxels.get(str(subject_info["t2_voxel_size"]), 0) + 1
        mask_voxels[str(subject_info["mask_voxel_size"])] = mask_voxels.get(str(subject_info["mask_voxel_size"]), 0) + 1

    dims["flair"] = flair_dims
    dims["t1"] = t1_dims
    dims["t2"] = t2_dims
    dims["mask"] = mask_dims

    voxels["flair"] = flair_voxels
    voxels["t1"] = t1_voxels
    voxels["t2"] = t2_voxels
    voxels["mask"] = mask_voxels

    orientations["flair"] = flair_orientation
    orientations["t1"] = t1_orientation
    orientations["t2"] = t2_orientation
    orientations["mask"] = mask_orientation

    output["dims"] = dims
    output["orientations"] = orientations
    output["voxels"] = voxels

    
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(output, f, indent=4)

    
        

if __name__ == "__main__":
    main()
    print(f"Saved EDA results to: {OUTPUT_JSON_PATH}")
    


