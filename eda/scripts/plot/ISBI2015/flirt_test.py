import nibabel as nib 
import os
import numpy as np
import matplotlib.pyplot as plt
import random

DATASET_ROOT = "/home/darshan/MS/data/ISBI2015/train"
OUTPUT_PATH = "/home/darshan/MS/eda/plots/ISBI2015/raw"


# Plot the mask, T1w, T2w, FLAIR IMAGE (random)
# Plot the mask, one of images of all dimensions for each class.

def get_img_dims(root, patient, time_point, mode):
    img_path = os.path.join(root, "orig", f"training{patient}_{time_point}_{mode.lower()}.nii.gz")
    if mode == "mask":
        img_path = os.path.join(root, "masks", f"training{patient}_{time_point}_mask1.nii")
    
    img = nib.load(img_path)
    return img, img.shape

def extract_middle_slice(img):

    # middle_idx = img.get_fdata().shape[2]//2 # axial view
    # middle_slice = f[:, :, middle_idx]
    # return middle_slice, middle_idx, f.shape, img.header.get_zooms()
    
    data = img.get_fdata()
    middle_idx = data.shape[2]//2
    middle_slice = data[:, :, middle_idx]

    return middle_slice, middle_idx, data.shape, img.header.get_zooms()


def main():
    flair_path = "/home/darshan/MS/preprocessing/output/flair/flair.nii.gz"
    t1_path = "/home/darshan/MS/preprocessing/output/t1/t1.nii.gz"
    t2_path = "/home/darshan/MS/preprocessing/output/t2/t2.nii.gz"
    mask_path = "/home/darshan/MS/preprocessing/output/mask/mask.nii.gz"

    flair_img = nib.load(flair_path)
    t1_img = nib.load(t1_path)
    t2_img = nib.load(t2_path)
    mask_img = nib.load(mask_path)

    flair_shape = flair_img.get_fdata().shape
    t1_shape = t1_img.get_fdata().shape
    t2_shape = t2_img.get_fdata().shape
    mask_shape = mask_img.get_fdata().shape

    filename = os.path.join("/home/darshan/MS/preprocessing/output/output.png")
    fig, axes = plt.subplots(2, 2, figsize=(20, 25))
    fig.suptitle(f"FLAIR, T1w, T2w and MASK plot Middle Slice taken from dimension 0 after transpose (1, 0, 2) ", fontsize=20)

    img_affine = flair_img.affine 
    mask_affine = mask_img.affine

    fig.suptitle(f"FLAIR, T1w, T2w and MASK after Resampling testing05_03\n Middle Slice taken from dimension 2 ", fontsize=20)

    flair_slice, flair_shape, flair_idx, flair_voxel_size = extract_middle_slice(flair_img)
    t1_slice, t1_shape, t1_idx, t1_voxel_size = extract_middle_slice(t1_img)
    t2_slice, t2_shape, t2_idx, t2_voxel_size = extract_middle_slice(t2_img)
    mask_slice, mask_shape, mask_idx, mask_voxel_size = extract_middle_slice(mask_img)

    axes[0, 0].imshow(flair_slice, cmap='gray')
    axes[0, 0].set_title(f"FLAIR DMENSIONS: {flair_shape}\nMIDDLE SLICE: {flair_idx} VOXEL SIZE: {flair_voxel_size}")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(t1_slice, cmap='gray')
    axes[0, 1].set_title(f"T1w DIMENSIONS: {t1_shape}\nMIDDLE SLICE: {t1_idx} VOXEL SIZE: {t1_voxel_size}")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(t2_slice, cmap='gray')
    axes[1, 0].set_title(f"T2w DIMENSIONS: {t2_shape}\nMIDDLE SLICE: {t2_idx} VOXEL SIZE: {t2_voxel_size}")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask_slice, cmap='gray')
    axes[1, 1].set_title(f"MASK DIMENSIONS: {mask_shape}\nMIDDLE SLCIE: {mask_idx} VOXEL SIZE: {mask_voxel_size}")
    axes[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


if __name__ == "__main__":
    main()


"""
python registration.py -i /home/darshan/MS/data/ISBI2015/train/training05/orig/training05_03_flair.nii.gz -t /home/darshan/MS/data/TEMPLATE/MNI152_T1_1mm.nii.gz -o /home/darshan/MS/preprocessing/output/flair/flair.nii.gz
&&
python registration.py -i /home/darshan/MS/data/ISBI2015/train/training05/orig/training05_03_mprage.nii.gz -t /home/darshan/MS/data/TEMPLATE/MNI152_T1_1mm.nii.gz -o /home/darshan/MS/preprocessing/output/t1/t1.nii.gz
&&
python registration.py -i /home/darshan/MS/data/ISBI2015/train/training05/orig/training05_03_t2.nii.gz -t /home/darshan/MS/data/TEMPLATE/MNI152_T1_1mm.nii.gz -o /home/darshan/MS/preprocessing/output/t2/t2.nii.gz
&&
python registration.py -i /home/darshan/MS/data/ISBI2015/train/training05/masks/training05_03_mask1.nii -t /home/darshan/MS/data/TEMPLATE/MNI152_T1_1mm.nii.gz -o /home/darshan/MS/preprocessing/output/mask/mask.nii.gz

"""