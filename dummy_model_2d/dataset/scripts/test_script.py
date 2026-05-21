import os
import glob
import shutil
import random
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from collections import Counter

# ---------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------
PREPROC_DIR = "/home/darshan/MS/dummy_model_2d/dataset/compiled/preprocessed"
PLOTS_DIR = "/home/darshan/MS/dummy_model_2d/dataset/compiled/plots"
REJECTED_DIR = "/home/darshan/MS/dummy_model_2d/dataset/compiled/rejected_mismatched"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REJECTED_DIR, exist_ok=True)

def run_eda():
    print("Starting Dataset Cleaning and EDA...\n")
    
    # ---------------------------------------------------------
    # Part 1: Filter Out Mismatched Data
    # ---------------------------------------------------------
    print("Scanning for dimension mismatches...")
    raw_flairs = sorted(glob.glob(os.path.join(PREPROC_DIR, "*_FLAIR.nii.gz")))
    
    rejected_count = 0
    valid_flairs = []
    
    for flair_path in raw_flairs:
        base_name = os.path.basename(flair_path).replace("_FLAIR.nii.gz", "")
        mask_path = os.path.join(PREPROC_DIR, f"{base_name}_MASK.nii.gz")
        
        if os.path.exists(mask_path):
            # Fast header load to check shapes without clogging RAM
            flair_shape = nib.load(flair_path).header.get_data_shape()
            mask_shape = nib.load(mask_path).header.get_data_shape()
            
            if flair_shape != mask_shape:
                print(f"  [REJECTED] {base_name}: FLAIR {flair_shape} vs MASK {mask_shape}")
                # Move both files to the rejected folder
                shutil.move(flair_path, os.path.join(REJECTED_DIR, os.path.basename(flair_path)))
                shutil.move(mask_path, os.path.join(REJECTED_DIR, os.path.basename(mask_path)))
                rejected_count += 1
            else:
                valid_flairs.append(flair_path)
        else:
            # If a mask is completely missing, also reject the flair
            print(f"  [REJECTED] {base_name}: Missing MASK file.")
            shutil.move(flair_path, os.path.join(REJECTED_DIR, os.path.basename(flair_path)))
            rejected_count += 1

    print(f"\nCleaning Complete. Filtered out {rejected_count} corrupted subjects.")
    print(f"Remaining valid training subjects: {len(valid_flairs)}\n")

    # ---------------------------------------------------------
    # Part 2: Plot 8 Random Samples (Guaranteed Valid Data)
    # ---------------------------------------------------------
    if len(valid_flairs) == 0:
        print("Error: No valid data left to plot!")
        return
        
    print("Generating EDA plotting grid...")
    if len(valid_flairs) < 8:
        sample_paths = valid_flairs
    else:
        sample_paths = random.sample(valid_flairs, 8)
        
    fig, axes = plt.subplots(nrows=8, ncols=2, figsize=(10, 24))
    # Handle the case where we have less than 8 valid samples gracefully
    if len(sample_paths) == 1: axes = np.expand_dims(axes, axis=0)
    
    fig.suptitle("Random Valid Samples: FLAIR vs MASK", fontsize=16)
    
    for i, flair_path in enumerate(sample_paths):
        base_name = os.path.basename(flair_path).replace("_FLAIR.nii.gz", "")
        mask_path = os.path.join(PREPROC_DIR, f"{base_name}_MASK.nii.gz")
        
        flair_data = nib.load(flair_path).get_fdata()
        mask_data = nib.load(mask_path).get_fdata()
        
        # Smart slice selection: find the slice with the most lesion pixels
        lesion_counts = np.sum(mask_data, axis=(0, 1))
        best_slice = np.argmax(lesion_counts)
        if lesion_counts[best_slice] == 0:
            best_slice = flair_data.shape[2] // 2  # Fallback to middle slice
            
        # Plot FLAIR
        ax_flair = axes[i, 0]
        ax_flair.imshow(flair_data[:, :, best_slice].T, cmap="gray", origin="lower")
        ax_flair.set_title(f"{base_name} - FLAIR (Slice {best_slice})")
        ax_flair.axis("off")
        
        # Plot MASK
        ax_mask = axes[i, 1]
        ax_mask.imshow(mask_data[:, :, best_slice].T, cmap="magma", origin="lower")
        ax_mask.set_title(f"{base_name} - MASK")
        ax_mask.axis("off")
        
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "samples.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved sample grid to {plot_path}")

    # ---------------------------------------------------------
    # Part 3: Verify Final Resampling Dimensions (Hashmap/Table)
    # ---------------------------------------------------------
    print("\nVerifying final clean dataset dimensions (X, Y)...")
    dimension_hashmap = Counter()
    
    for flair_path in valid_flairs:
        shape = nib.load(flair_path).header.get_data_shape()
        xy_dims = (shape[0], shape[1])
        dimension_hashmap[xy_dims] += 1
        
    print(f"\n{'-'*45}")
    print(f"{'Spatial Dimensions (X, Y)':<30} | {'Count'}")
    print(f"{'-'*45}")
    for dims, count in dimension_hashmap.items():
        print(f"{str(dims):<30} | {count}")
    print(f"{'-'*45}")
    print(f"Total training pairs verified: {len(valid_flairs)}\n")

if __name__ == "__main__":
    run_eda()