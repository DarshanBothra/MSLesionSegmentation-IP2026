# EDA For ISBI2015 dataset
# Data for 4 time points for a single patient

import os
import nibabel as nib
import json
import random
import numpy as np

DATASET_ROOT = "/home/darshan/MS/data/RAW/ISBI2015"
TRAIN_ROOT = os.path.join(DATASET_ROOT, "train")
TEST_ROOT = os.path.join(DATASET_ROOT, "test") # cannot be used, no mask present!
OUTPUT_JSON_PATH = "/home/darshan/MS/eda/results/exploratory/isbi_eda.json"

# Explore what all kinds of images are present (T1w, T2w, FLAIR, etc.)
def explore_images(root, patient, time_point, required):
    """Explore the dataset and print the different types of images present.
    Args: root - Path to original/preprocessed images for a patient"""
    types = set()
    PATIENT_IMAGES = os.path.join(root, f"training{patient}", "orig")
    files = os.listdir(PATIENT_IMAGES) # naming: trainingX_X_flair.nii.gz
    for file in files:
        mode = file.split("_")[-1].replace(".nii.gz", "")
        types.add(mode)

    record = {}
    missing = []
    present = []
    for req in required:
        if req not in types:
            missing.append(req)
        else:
            present.append(req)
    record["present_count"] = len(present)
    record["present"] = present
    record["missing_count"] = len(missing)
    record["missing"] = missing
    record["all_present"] = len(missing) == 0
    record["all_missing"] = len(present) == 0

    return record

def count_time_samples(root, patient, mode):
    """
    Counts the number of time points for which a particular mode (mprage, flair, t2w) of images are present
    Args: 
        root - root training directory
        patient - subject number
        mode - mparage (t1w), flair, t2w
    """
    mode = mode.lower()
    PATIENT_DIR = os.path.join(root, f"training{patient}", "orig")
    for file in os.listdir(PATIENT_DIR):
        if mode in file.lower():
            count +=1
    return count

def count_subjects(root):
    """Counts the number of subjects in the dataset.
    Args: root - path to dataset patient folders"""
    count = 0 # number of training subjects
    for file in os.listdir(root):
        if (os.path.isdir(os.path.join(root, file))):
            count +=1
    return count

def is_mask_present(root, patient, time_point):
    """ Check if the mask is present in the directory or not
    Args: root - path to dataset patient folders where masks are supposed to be present
    """
    
    # Mask is of the form training(patient)_time_point_mask1.nii.gz
    if (os.path.exists(os.path.join(root, f"training{patient}", "masks", f"training{patient}_{time_point}_mask1.nii"))):
        return True
    else:
        return False
    
def get_img_dims(root, patient, time_point, mode):
    """
    root = training root
    """
    IMG_PATH = os.path.join(root, f"training{patient}", "orig", f"training{patient}_{time_point}_{mode.lower()}.nii.gz")
    if mode == "mask":
        IMG_PATH = os.path.join(root, f"training{patient}", "masks", f"training{patient}_{time_point}_mask1.nii")
    img = nib.load(IMG_PATH)
    img = nib.as_closest_canonical(img)
    orientation = nib.aff2axcodes(img.affine)
    return img.shape, orientation

def eda():

    output = {}

    subjects = count_subjects(TRAIN_ROOT)
    output["total_subjects"] = subjects

    time_points = ["01", "02", "03", "04"]
    patients = ["01", "02", "03", "04", "05"]

    # Set all to true if missing set to false
    output["all_flair_present"] = True
    output["all_t1w_present"] = True
    output["all_t2w_present"] = True
    output["all_masks_present"] = True
    output["total_flair"] = 0
    output["total_t1w"] = 0
    output["total_t2w"] = 0
    output["total_mask"] = 0

    # Patient wise reocrds
    pw_record = {}
    for patient in patients:
        patient_record = {}
        for tp in time_points:
            tp_record = {}
            stats = explore_images(TRAIN_ROOT, patient, tp, ["mprage", "flair", "t2"])
            tp_record.update(stats)
            tp_record["mask_present"] = is_mask_present(TRAIN_ROOT, patient, tp)
            if tp_record["mask_present"]:
                output["total_mask"] += 1
            if not tp_record["mask_present"]:
                output["all_masks_present"] = False
            patient_record[tp] = tp_record
        
        flair_present = []
        flair_missing = []
        for tp in time_points:
            if "flair" in patient_record[tp]["present"]:
                flair_present.append(tp)
                output["total_flair"] += 1
            else:
                flair_missing.append(tp)
        if len(flair_missing) != 0:
            output["all_flair_present"] = False
        patient_record["flair_present"] = flair_present
        patient_record["total_flair_present"] = len(flair_present)
        patient_record["flair_missing"] = flair_missing
        patient_record["total_flair_missing"] = len(flair_missing)

        t1w_present = []
        t1w_missing = []
        for tp in time_points:
            if "mprage" in patient_record[tp]["present"]:
                t1w_present.append(tp)
                output["total_t1w"] += 1
            else:
                t1w_missing.append(tp)
        if len(t1w_missing) != 0:
            output["all_t1w_present"] = False
        patient_record["t1w_present"] = t1w_present
        patient_record["total_t1w_present"] = len(t1w_present)
        patient_record["t1w_missing"] = t1w_missing
        patient_record["total_t1w_missing"] = len(t1w_missing)


        t2w_present = []
        t2w_missing = []
        for tp in time_points:
            if "t2" in patient_record[tp]["present"]:
                t2w_present.append(tp)
                output["total_t2w"] += 1
            else:
                t2w_missing.append(tp)
        if len(t2w_missing) != 0:
            output["all_t2w_present"] = False
        patient_record["t2w_present"] = t2w_present
        patient_record["total_t2w_present"] = len(t2w_present)
        patient_record["t2w_missing"] = t2w_missing
        patient_record["total_t2w_missing"] = len(t2w_missing)

        pw_record[patient] = patient_record
    output["patient_wise"] = pw_record

    # Hashmap for image dimensions for all images (modality wise): total it out later
    dims = {}
    modes = ["flair", "mprage", "t2"]
    total_dims = {}
    mask_dims = {}
    for mode in modes:
        mode_dims = {}
        for patient in patients:
            for tp in time_points:
                if mode in patient_record[tp]["present"]:
                    img_dims, orientation = get_img_dims(TRAIN_ROOT, patient, tp, mode)
                    img_dims, orientation = str(img_dims), str(orientation)
                    if img_dims in mode_dims:
                        mode_dims[img_dims] += 1
                    else:
                        mode_dims[img_dims] = 1
                    if img_dims in total_dims:
                        total_dims[img_dims] +=1
                    else:
                        total_dims[img_dims] = 1
        dims[mode] = mode_dims
    
    for patient in patients:
        for tp in time_points:
            if output["patient_wise"][patient][tp]["mask_present"]:
                img_dims, orientation = get_img_dims(TRAIN_ROOT, patient, tp, "mask")
                img_dims, orientation = str(img_dims), str(orientation)
                if img_dims in mask_dims:
                    mask_dims[img_dims] += 1
                else:
                    mask_dims[img_dims] = 1

    dims["mask"] = mask_dims
    dims["combined"] = total_dims
    output["dimensions"] = dims


    # dump to JSON
    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(output, f, indent=4)

def main():
    eda()
    print(f"EDA Saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()