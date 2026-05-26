import os
import nibabel as nib
import numpy as np
import json
import random

DATASET_ROOT = "/home/darshan/MS/data/MICCAI2016"
TRAIN_ROOT = os.path.join(DATASET_ROOT, "Training")
TEST_ROOT = os.path.join(DATASET_ROOT, "Testing")
TRAINING_CENTERS = [x for x in os.listdir(TRAIN_ROOT) if os.path.isdir(os.path.join(TRAIN_ROOT, x))]
TESTING_CENTERS = [x for x in os.listdir(TEST_ROOT) if os.path.isdir(os.path.join(TEST_ROOT, x))]
OUTPUT_JSON_PATH = "/home/darshan/MS/eda/results/miccai2016_eda.json"

def explore_images(root, center, patient, required):
    types = set()
    PATIENT_PATH = os.path.join(root, center, patient, "Raw_Data")
    files = os.listdir(PATIENT_PATH)
    for file in files:
        mode = file.split("_")[-1].replace(".nii.gz", "")
        if mode in required:
            types.add(mode)
    record = {}
    missing = []
    present = []
    for req in required:
        if req in types:
            present.append(req)
        else:
            missing.append(req)
    
    record["present"] = present
    record["present_count"] = len(present)
    record["missing"] = missing
    record["missing_count"] = len(missing)
    record["all_present"] = len(missing) == 0
    record["all_missing"] = len(present) == 0
        
    return record

def is_mask_present(root, center, patient):
    return os.path.exists(os.path.join(root, center, patient, "Masks", "Consensus.nii.gz"))

def get_image_dims(root, center, patient, mode):
    if mode == "MASK":
        IMG_PATH = os.path.join(root, center, patient, "Masks", "Consensus.nii.gz")
    else:
        IMG_PATH = os.path.join(root, center, patient, "Raw_Data", f"{mode}.nii.gz")

    if not os.path.exists(IMG_PATH):
        return None

    img = nib.load(IMG_PATH)
    return img.shape

def eda():

    output = {}
    output["total_subjects"] = 0

    output["all_flair_present"] = True 
    output["all_t1w_present"] = True
    output["all_t2w_present"] = True
    output["all_masks_present"] = True
    
    output["total_flair"] = 0
    output["total_t1w"] = 0
    output["total_t2w"] = 0
    output["total_mask"] = 0

    # EDA for training images
    training_out = {}
    training_out["subjects"] = 0
    for center in TRAINING_CENTERS:
        patients = [x for x in os.listdir(os.path.join(TRAIN_ROOT, center)) if os.path.isdir(os.path.join(TRAIN_ROOT, center, x))]
        center_wise = {}
        center_wise["subjects"] = len(patients)
        training_out["subjects"] += len(patients)
        output["total_subjects"] += len(patients)
        for patient in patients:
            # conduct some patient eda
            patient_record = {}
            stats = explore_images(TRAIN_ROOT, center, patient, ["FLAIR", "T1", "T2"])
            patient_record.update(stats)
            patient_record["mask_present"] = is_mask_present(TRAIN_ROOT, center, patient)
            if patient_record["mask_present"]:
                output["total_mask"] += 1
            else:
                output["all_masks_present"] = False
            center_wise[patient] = patient_record
        
        flair_present = [] # for what patients in the center flair is present
        flair_missing = [] # for what patients in the center flair is missing
        for patient in patients:
            if "FLAIR" in center_wise[patient]["present"]:
                flair_present.append(patient)
                output["total_flair"] += 1
            else:
                flair_missing.append(patient)
        if len(flair_missing) != 0:
            output["all_flair_present"] = False
        center_wise["flair_present"] = flair_present
        center_wise["total_flair_present"] = len(flair_present)
        center_wise["flair_missing"] = flair_missing
        center_wise["total_flair_missing"] = len(flair_missing)

        t1w_present = [] # for what patients in the center t1w is present
        t1w_missing = [] # for what patients in the center t1w is missing
        for patient in patients:
            if "T1" in center_wise[patient]["present"]:
                t1w_present.append(patient)
                output["total_t1w"] += 1
            else:
                t1w_missing.append(patient)
        if len(t1w_missing) != 0:
            output["all_t1w_present"] = False
        center_wise["t1w_present"] = t1w_present
        center_wise["total_t1w_present"] = len(t1w_present)
        center_wise["t1w_missing"] = t1w_missing
        center_wise["total_t1w_missing"] = len(t1w_missing)

        t2w_present = [] # for what patients in the center t1w is present
        t2w_missing = [] # for what patients in the center t1w is missing
        for patient in patients:
            if "T2" in center_wise[patient]["present"]:
                t2w_present.append(patient)
                output["total_t2w"] += 1
            else:
                t2w_missing.append(patient)
        if len(t2w_missing) != 0:
            output["all_t2w_present"] = False
        center_wise["t2w_present"] = t2w_present
        center_wise["total_t2w_present"] = len(t2w_present)
        center_wise["t2w_missing"] = t2w_missing
        center_wise["total_t2w_missing"] = len(t2w_missing)

        training_out[center] = center_wise
    output["training"] = training_out
    
    # EDA for testing images
    testing_out = {}
    testing_out["subjects"] = 0
    for center in TESTING_CENTERS:
        patients = [x for x in os.listdir(os.path.join(TEST_ROOT, center)) if os.path.isdir(os.path.join(TEST_ROOT, center, x))]
        center_wise = {}
        center_wise["subjects"] = len(patients)
        testing_out["subjects"] += len(patients)
        output["total_subjects"] += len(patients)
        for patient in patients:
            # conduct some patient eda
            patient_record = {}
            stats = explore_images(TEST_ROOT, center, patient, ["FLAIR", "T1", "T2"])
            patient_record.update(stats)
            patient_record["mask_present"] = is_mask_present(TEST_ROOT, center, patient)
            if patient_record["mask_present"]:
                output["total_mask"] += 1
            else:
                output["all_masks_present"] = False
            center_wise[patient] = patient_record
        
        flair_present = [] # for what patients in the center flair is present
        flair_missing = [] # for what patients in the center flair is missing
        for patient in patients:
            if "FLAIR" in center_wise[patient]["present"]:
                flair_present.append(patient)
                output["total_flair"] += 1
            else:
                flair_missing.append(patient)
        if len(flair_missing) != 0:
            output["all_flair_present"] = False
        center_wise["flair_present"] = flair_present
        center_wise["total_flair_present"] = len(flair_present)
        center_wise["flair_missing"] = flair_missing
        center_wise["total_flair_missing"] = len(flair_missing)

        t1w_present = [] # for what patients in the center t1w is present
        t1w_missing = [] # for what patients in the center t1w is missing
        for patient in patients:
            if "T1" in center_wise[patient]["present"]:
                t1w_present.append(patient)
                output["total_t1w"] += 1
            else:
                t1w_missing.append(patient)
        if len(t1w_missing) != 0:
            output["all_t1w_present"] = False
        center_wise["t1w_present"] = t1w_present
        center_wise["total_t1w_present"] = len(t1w_present)
        center_wise["t1w_missing"] = t1w_missing
        center_wise["total_t1w_missing"] = len(t1w_missing)

        t2w_present = [] # for what patients in the center t1w is present
        t2w_missing = [] # for what patients in the center t1w is missing
        for patient in patients:
            if "T2" in center_wise[patient]["present"]:
                t2w_present.append(patient)
                output["total_t2w"] += 1
            else:
                t2w_missing.append(patient)
        if len(t2w_missing) != 0:
            output["all_t2w_present"] = False
        center_wise["t2w_present"] = t2w_present
        center_wise["total_t2w_present"] = len(t2w_present)
        center_wise["t2w_missing"] = t2w_missing
        center_wise["total_t2w_missing"] = len(t2w_missing)

        testing_out[center] = center_wise
    output["testing"] = testing_out

    # Hashmap for image dimensions for all images modality wise!
    dims = {}
    modes = ["FLAIR", "T1", "T2", "MASK"]
    total_dims = {}
    for mode in modes:
        mode_dims = {}
        for centers in [TRAINING_CENTERS, TESTING_CENTERS]:
            for center in centers:
                patients = [x for x in os.listdir(os.path.join(TRAIN_ROOT, center)) if os.path.isdir(os.path.join(TRAIN_ROOT, center, x))] if centers == TRAINING_CENTERS else [x for x in os.listdir(os.path.join(TEST_ROOT, center)) if os.path.isdir(os.path.join(TEST_ROOT, center, x))]
                for patient in patients:
                    batch = "training" if centers == TRAINING_CENTERS else "testing"
                    root = TRAIN_ROOT if centers == TRAINING_CENTERS else TEST_ROOT
                    if mode in output[batch][center][patient]["present"]:
                        img_dims = get_image_dims(root, center, patient, mode)
                        img_dims = str(img_dims)
                        if img_dims in mode_dims:
                            mode_dims[img_dims] += 1
                            total_dims[img_dims] += 1
                        else:
                            mode_dims[img_dims] = 1
                            total_dims[img_dims] = 1
                        
                    elif mode == "MASK":
                        img_dims = get_image_dims(root, center, patient, mode)
                        img_dims = str(img_dims)
                        if img_dims in mode_dims:
                            mode_dims[img_dims] += 1
                        else:
                            mode_dims[img_dims] = 1
        dims[mode.lower()] = mode_dims
    output["dimensions"] = dims
    output["dimensions"]["combined"] = total_dims

    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(output, f, indent=4)

def main():
    eda()
    print(f"EDA saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()