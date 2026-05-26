import os
import nibabel as nib
import numpy as np
import json
import random

DATASET_ROOT = "/home/darshan/MS/data/MSLegSeg"
TRAIN_ROOT = os.path.join(DATASET_ROOT, "MSLegSeg_RAW" )
MASK_ROOT = os.path.join(DATASET_ROOT, "MSLegSeg Dataset", "train")
TEST_ROOT = os.path.join(DATASET_ROOT, "MSLegSeg Dataset", "test")
OUTPUT_JSON_PATH = "/home/darshan/MS/eda/results/msleg_eda.json"

def explore_images(root, patient, time_point, required):
    types = set()
    PATIENT_PATH = os.path.join(root, patient, time_point)
    files = os.listdir(PATIENT_PATH)
    for file in files:
        mode = file.split("_")[-1].replace(".nii.gz", "")
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

def count_subjects(root):
    return len(os.listdir(root))

def is_mask_present(root, patient, time_point):
    """ Check if the mask is present in the directory or not
    Args: root - path to dataset patient folders where masks are supposed to be present
    """
    
    # Mask is of the form training(patient)_time_point_mask1.nii.gz
    if (os.path.exists(os.path.join(root, patient, time_point, f"{patient}_{time_point}_MASK.nii.gz"))):
        return True
    elif (os.path.exists(os.path.join(TEST_ROOT, patient, f"{patient}_MASK.nii.gz"))):
        return True
    else:
        return False
    
def get_img_dims(root, patient, time_point, mode):
    """
    root = training root
    """
    IMG_PATH = os.path.join(root, patient, time_point, f"{patient}_{time_point}_{mode.upper()}.nii.gz")
    if not os.path.exists(IMG_PATH):
        IMG_PATH = os.path.join(TEST_ROOT, patient, f"{patient}_{mode.upper()}.nii.gz")
    img = nib.load(IMG_PATH)
    return img.shape

def eda():
    output = {}
    subjects = count_subjects(TRAIN_ROOT)
    output["total_subjects"] = subjects

    output["all_flair_present"] = True
    output["all_t1w_present"] = True
    output["all_t2w_present"] = True
    output["all_masks_present"] = True

    output["total_flair"] = 0
    output["total_t1w"] = 0
    output["total_t2w"] = 0
    output["total_mask"] = 0

    patients = sorted([os.path.basename(x) for x in os.listdir(TRAIN_ROOT)]) # Px
    
    pw_record = {}
    for patient in patients:
        patient_record = {}
        time_points = os.listdir(os.path.join(TRAIN_ROOT, patient))
        time_points = sorted(time_points)
        for tp in time_points:
            tp_record = {}
            stats = explore_images(TRAIN_ROOT, patient, tp, ["FLAIR", "T1", "T2"])
            tp_record.update(stats)
            tp_record["mask_present"] = is_mask_present(MASK_ROOT, patient, tp)
            if tp_record["mask_present"]:
                output["total_mask"] += 1
            if not tp_record["mask_present"]:
                output["all_masks_present"] = False
            patient_record[tp] = tp_record
        
        flair_present = []
        flair_missing = []
        for tp in time_points:
            if "FLAIR" in patient_record[tp]["present"]:
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
            if "T1" in patient_record[tp]["present"]:
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
            if "T2" in patient_record[tp]["present"]:
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
    modes = ["FLAIR", "T1", "T2", "MASK"]
    total_dims = {}
    mask_dims = {}
    for mode in modes:
        mode_dims = {}
        for patient in patients:
            time_points = os.listdir(os.path.join(TRAIN_ROOT, patient))
            time_points = sorted(time_points)

            for tp in time_points:
                if mode in output["patient_wise"][patient][tp]["present"]:
                    img_dims = get_img_dims(TRAIN_ROOT, patient, tp, mode) if mode != "MASK" else get_img_dims(MASK_ROOT, patient, tp, mode)
                    img_dims = str(img_dims)
                    if img_dims in mode_dims:
                        mode_dims[img_dims] += 1
                    else:
                        mode_dims[img_dims] = 1
                    if img_dims in total_dims:
                        total_dims[img_dims] +=1
                    else:
                        total_dims[img_dims] = 1
        dims[mode.lower()] = mode_dims
    
    for patient in patients:
        time_points = os.listdir(os.path.join(TRAIN_ROOT, patient))
        time_points = sorted(time_points)
        for tp in time_points:
            if output["patient_wise"][patient][tp]["mask_present"]:
                img_dims = get_img_dims(MASK_ROOT, patient, tp, "MASK")
                img_dims = str(img_dims)
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
    




