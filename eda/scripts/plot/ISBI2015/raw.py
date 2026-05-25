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
    patients = [x[-2:] for x in os.listdir(DATASET_ROOT) if x.startswith("training") and os.path.isdir(os.path.join(DATASET_ROOT, x))]
    patients = sorted(patients)
    time_points = ["01", "02", "03", "04"]
    modes = ["flair", "mprage", "t2", "mask"]

    data = {} # all data for ISBI2015 database patient, timepoint wise

    for patient in patients:
        patient_record = {}
        for tp in time_points:
            tp_record = {}
            patient_root = os.path.join(DATASET_ROOT, f"training{patient}")
            
            for mode in modes:
                img, dims = get_img_dims(patient_root, patient, tp, mode)
                mode = "t1" if mode == "mprage" else mode
                tp_record[mode] = img
                tp_record[f"{mode}_dims"] = dims
            
            patient_record[tp] = tp_record
        data[patient] = patient_record

    # Choose random timepoints for all 5 patients, print t1, t2, flair, mask for them and dimensions as well

    time_samples = {} # key: patient, value: time_point
    np.random.seed(42) # for reproducability
    for i in range(1, 6):
        time_samples[f"0{i}"] = np.random.choice(time_points)
    # Plot
    for patient, tp in time_samples.items():
        filename = os.path.join(OUTPUT_PATH, f"training{patient}_{tp}_raw.png")
        fig, axes = plt.subplots(2, 2, figsize=(20, 25))
        fig.suptitle(f"FLAIR, T1w, T2w and MASK plot for Patient:{patient} Time Point: {tp}\n Middle Slice taken from dimension 2 ", fontsize=20)

        flair_slice, flair_shape, flair_idx, flair_voxel_size = extract_middle_slice(data[patient][tp]["flair"])
        t1_slice, t1_shape, t1_idx, t1_voxel_size = extract_middle_slice(data[patient][tp]["t1"])
        t2_slice, t2_shape, t2_idx, t2_voxel_size = extract_middle_slice(data[patient][tp]["t2"])
        mask_slice, mask_shape, mask_idx, mask_voxel_size = extract_middle_slice(data[patient][tp]["mask"])

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



