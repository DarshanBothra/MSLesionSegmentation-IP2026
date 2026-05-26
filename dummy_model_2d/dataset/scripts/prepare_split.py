import os
import glob
import random
import shutil

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PREPROC_DIR = "/home/darshan/MS/dummy_model_2d/dataset/compiled/preprocessed"
SPLIT_BASE_DIR = "/home/darshan/MS/dummy_model_2d/dataset/split"

TRAIN_DIR = os.path.join(SPLIT_BASE_DIR, "train")
VAL_DIR = os.path.join(SPLIT_BASE_DIR, "val")
TEST_DIR = os.path.join(SPLIT_BASE_DIR, "test")

print(len(os.listdir(TRAIN_DIR)))
print(len(os.listdir(VAL_DIR)))
print(len(os.listdir(TEST_DIR)))
exit()

# ---------------------------------------------------------
# Execution
# ---------------------------------------------------------
def prepare_split():
    print("Preparing train/val/test dataset split...")
    
    # Create the destination directories if they don't exist
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(split_dir, exist_ok=True)
        
    # Gather all valid pairs (only grabbing FLAIRs that have a matching MASK)
    raw_flairs = sorted(glob.glob(os.path.join(PREPROC_DIR, "*_FLAIR.nii.gz")))
    valid_pairs = []
    
    for flair_path in raw_flairs:
        mask_path = flair_path.replace("_FLAIR.nii.gz", "_MASK.nii.gz")
        if os.path.exists(mask_path):
            valid_pairs.append((flair_path, mask_path))
            
    total_subjects = len(valid_pairs)
    if total_subjects == 0:
        print(f"Error: No valid FLAIR/MASK pairs found in {PREPROC_DIR}")
        return

    # Shuffle the dataset deterministically
    random.seed(1100)  # Ensures the same split every time you run this script
    random.shuffle(valid_pairs)
    
    # Calculate indices for an 80 / 10 / 10 split
    train_end = int(total_subjects * 0.8)
    val_end = int(total_subjects * 0.9)
    
    train_pairs = valid_pairs[:train_end]
    val_pairs = valid_pairs[train_end:val_end]
    test_pairs = valid_pairs[val_end:]
    
    # Helper function to copy files
    def copy_split(pairs, dest_folder, split_name):
        print(f"Copying {len(pairs)} subjects to {split_name}...")
        for flair_src, mask_src in pairs:
            # Generate destination paths
            flair_dest = os.path.join(dest_folder, os.path.basename(flair_src))
            mask_dest = os.path.join(dest_folder, os.path.basename(mask_src))
            
            # Copy the files
            shutil.copy(flair_src, flair_dest)
            shutil.copy(mask_src, mask_dest)

    # Execute copies
    copy_split(train_pairs, TRAIN_DIR, "TRAIN")
    copy_split(val_pairs, VAL_DIR, "VAL")
    copy_split(test_pairs, TEST_DIR, "TEST")
    
    print("\nSplit Complete!")
    print(f"Total Subjects: {total_subjects}")
    print(f"Train: {len(train_pairs)} (80%) | Val: {len(val_pairs)} (10%) | Test: {len(test_pairs)} (10%)")
    print(f"Files saved to {SPLIT_BASE_DIR}")

if __name__ == "__main__":
    prepare_split()