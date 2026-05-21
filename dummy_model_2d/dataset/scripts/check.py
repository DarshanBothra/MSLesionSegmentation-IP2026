# Script to rename final datapoints as Sx for x in 0 to 100.

import os
ROOT = "/home/darshan/MS/dummy_model_2d/dataset/split"
TRAIN_ROOT = os.path.join(ROOT, "train")
VAL_ROOT = os.path.join(ROOT, "val")
TEST_ROOT = os.path.join(ROOT, "test")

def count_samples(train_path, val_path, test_path):
    train_samples = {}
    val_samples = {}
    test_samples = {}

    train_files = os.listdir(train_path)
    test_files = os.listdir(test_path)
    val_files = os.listdir(val_path)
    
    total_train_flair = 0
    total_train_mask = 0
    total_val_flair = 0
    total_val_mask = 0
    total_test_flair = 0
    total_test_mask = 0

    for file in train_files:
        if "FLAIR" in file:
            total_train_flair += 1
        if "MASK" in file:
            total_train_mask += 1
    
    for file in val_files:
        if "FLAIR" in file:
            total_val_flair += 1
        if "MASK" in file:
            total_val_mask += 1
    
    for file in test_files:
        if "FLAIR" in file:
            total_test_flair += 1
        if "MASK" in file:
            total_test_mask += 1

    if (total_train_flair != total_train_mask) or \
    (total_val_flair != total_val_mask) or \
    (total_test_flair != total_test_mask):
        print("Error: Number of FLAIR and MASK files do not match in one or more of the splits")
    else:
        print("All splits have equal number of FLAIR and MASK files")
        print("Total Train Samples: ", total_train_flair)
        print("Total Val Samples: ", total_val_flair)
        print("Total Test Samples: ", total_test_flair)

    for file in train_files:
        if "FLAIR" in file:
            case = file.split("_")[0]
            mask_file = case + "_MASK.nii.gz"
            if mask_file in train_files:
                train_samples[file] = mask_file
    
    for file in val_files:
        if "FLAIR" in file:
            case = file.split("_")[0]
            mask_file = case + "_MASK.nii.gz"
            if mask_file in val_files:
                val_samples[file] = mask_file

    for file in test_files:
        if "FLAIR" in file:
            case = file.split("_")[0]
            mask_file = case + "_MASK.nii.gz"
            if mask_file in test_files:
                test_samples[file] = mask_file
        

    return train_samples, val_samples, test_samples


if __name__ == "__main__":
    train_samples, val_samples, test_samples = count_samples(TRAIN_ROOT, VAL_ROOT, TEST_ROOT)

