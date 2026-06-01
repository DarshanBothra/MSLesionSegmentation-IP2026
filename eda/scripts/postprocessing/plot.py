import os
import nibabel as nib 
import matplotlib.pyplot as plt 
import numpy as np

SAMPLE_PATH = "/home/darshan/MS/data/PREPROCESSED/S3_FLIRT"

def extract_middle_slice(file):
    img = nib.load(file)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()

    # data = np.transpose(data, (1, 0, 2))
    print(data.shape)
    middle_idx = data.shape[2]//2
    slice_2d = data[:, : , middle_idx+30]

    return slice_2d, middle_idx, data.shape, img.header.get_zooms()

def main():
    output_file = "/home/darshan/MS/preprocessing/output/mslegseg_3_fsl.png"
    flair_file = os.path.join(SAMPLE_PATH, "flair_reg.nii.gz")
    t1_file = os.path.join(SAMPLE_PATH, "t1_reg.nii.gz")
    t2_file = os.path.join(SAMPLE_PATH, "t2_reg.nii.gz")
    mask_file = os.path.join(SAMPLE_PATH, "mask_reg.nii.gz")

    flair_slice, flair_idx, flair_shape, flair_voxel = extract_middle_slice(flair_file)
    t1_slice, t1_idx, t1_shape, t1_voxel = extract_middle_slice(t1_file)
    t2_slice, t2_idx, t2_shape, t2_voxel = extract_middle_slice(t2_file)
    mask_slice, mask_idx, mask_shape, mask_voxel = extract_middle_slice(mask_file)

    fig, axes = plt.subplots(2, 2, figsize=(20, 25))

    axes[0, 0].imshow(t1_slice, cmap='gray')
    axes[0, 0].set_title(f"Patient: S11\nT1w Preprocessed\nShape: {t1_shape}\nVoxel Size: {t1_voxel}\nMiddle Slice: {t1_idx}")
    axes[0, 0].axis('off')
        
    axes[0, 1].imshow(t2_slice, cmap='gray')
    axes[0, 1].set_title(f"Patient: S11\nT2w Preprocessed\nShape: {t2_shape}\nVoxel Size: {t2_voxel}\nMiddle Slice: {t2_idx}")
    axes[0, 1].axis('off')
        
    axes[1, 0].imshow(flair_slice, cmap='gray')
    axes[1, 0].set_title(f"Patient: S11\nFLAIR Preprocessed\nShape: {flair_shape}\nVoxel Size: {flair_voxel}\nMiddle Slice: {flair_idx}")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask_slice, cmap='gray')
    axes[1, 1].set_title(f"Patient: S11\nMask Preprocessed\nShape: {mask_shape}\nVoxel Size: {mask_voxel}\nMiddle Slice: {mask_idx}")
    axes[1, 1].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file)
    plt.close()
    print(f"Saved {output_file}")

    output_file = "/home/darshan/MS/preprocessing/output/mslegseg_3_original.png"
    flair_slice, flair_idx, flair_shape, flair_voxel = extract_middle_slice("/home/darshan/MS/data/RAW/MSLegSeg/MSLegSeg_RAW/P45/T1/P45_T1_FLAIR.nii.gz")
    t1_slice, t1_idx, t1_shape, t1_voxel = extract_middle_slice("/home/darshan/MS/data/RAW/MSLegSeg/MSLegSeg_RAW/P45/T1/P45_T1_T1.nii.gz")
    t2_slice, t2_idx, t2_shape, t2_voxel = extract_middle_slice("/home/darshan/MS/data/RAW/MSLegSeg/MSLegSeg_RAW/P45/T1/P45_T1_T2.nii.gz")
    mask_slice, mask_idx, mask_shape, mask_voxel = extract_middle_slice("/home/darshan/MS/data/RAW/MSLegSeg/MSLegSeg_Dataset/train/P45/T1/P45_T1_MASK.nii.gz")

    fig, axes = plt.subplots(2, 2, figsize=(20, 25))

    axes[0, 0].imshow(t1_slice, cmap='gray')
    axes[0, 0].set_title(f"Patient: S11\nT1w Preprocessed\nShape: {t1_shape}\nVoxel Size: {t1_voxel}\nMiddle Slice: {t1_idx}")
    axes[0, 0].axis('off')
        
    axes[0, 1].imshow(t2_slice, cmap='gray')
    axes[0, 1].set_title(f"Patient: S11\nT2w Preprocessed\nShape: {t2_shape}\nVoxel Size: {t2_voxel}\nMiddle Slice: {t2_idx}")
    axes[0, 1].axis('off')
        
    axes[1, 0].imshow(flair_slice, cmap='gray')
    axes[1, 0].set_title(f"Patient: S11\nFLAIR Preprocessed\nShape: {flair_shape}\nVoxel Size: {flair_voxel}\nMiddle Slice: {flair_idx}")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask_slice, cmap='gray')
    axes[1, 1].set_title(f"Patient: S11\nMask Preprocessed\nShape: {mask_shape}\nVoxel Size: {mask_voxel}\nMiddle Slice: {mask_idx}")
    axes[1, 1].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file)
    plt.close()
    print(f"Saved {output_file}")




if __name__ == "__main__":
    main()

