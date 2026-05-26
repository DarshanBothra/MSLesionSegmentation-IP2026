# MSLesSeg Dataset Exploratory Data Analysis (EDA)

The **MSLesSeg Dataset** contains two primary groupings of data within the workspace:
1. **MSLesSeg_RAW**: The original unstructured/raw scans.
2. **MSLesSeg Dataset**: The structured, preprocessed data split into `train` and `test` subsets, designed for model training.

---

## 1. MSLesSeg_RAW (Original Scans)

This directory contains the original scans for 75 patients without standardized preprocessing or masks. 

- **Total Patients:** 75 (Folders `P1` through `P75`)
- **Total Timepoints/Studies:** 118
- **Modalities Count:**
  - **T1:** 92 images
  - **T2:** 143 images
  - **FLAIR:** 118 images
- **Masks:** 0 masks found in this directory.
- **Image Dimensions:** The images come in various heterogeneous resolutions and spatial dimensions, including but not limited to:
  - `(512, 512, 188)`
  - `(256, 256, 44)`
  - `(1024, 1024, 100)`
  - `(672, 672, 22)`
  - `(364, 448, 224)`
- **Lesion vs. Control:** Since there are no masks provided in the RAW format, we rely on the preprocessed dataset to delineate the lesions. 

---

## 2. MSLesSeg Dataset (Preprocessed & Structured)

The main directory contains preprocessed data specifically split and prepared for machine learning models (like 3D/2D U-Net). The images have been resampled, registered, and padded/cropped to uniform dimensions.

### A. Training Set (`train/`)
The training set is organized by Patient and then Timepoint (e.g., `train/P1/T1/`).

- **Total Patients:** 53
- **Total Timepoints:** 93
- **Modalities Count:**
  - **T1:** 68 images
  - **T2:** 118 images
  - **FLAIR:** 93 images
- **Masks:** 93 masks (Every single timepoint has a corresponding lesion mask).
- **Image Dimensions:** All images and masks have been perfectly standardized to `(182, 218, 182)`.

### B. Testing Set (`test/`)
The test set is organized directly by Patient (e.g., `test/P54/`), with no `T` subfolders.

- **Total Patients:** 22
- **Modalities Count:**
  - **T1:** 22 images
  - **T2:** 22 images
  - **FLAIR:** 22 images
- **Masks:** 22 masks.
- **Image Dimensions:** All images and masks share the standardized `(182, 218, 182)` dimension.

---

## 3. Analysis of Controls vs. Lesion Cases

To evaluate whether a case is a control (healthy/no MS lesions) or a lesion case (patient with MS lesions), we analyzed the binary ground truth `MASK.nii.gz` arrays to see if any voxel was labeled as a lesion (`> 0`).

- **Train Set Lesion Cases:** 93 (100% of the train timepoints have $>0$ voxels in their mask)
- **Train Set Control Cases:** 0 
- **Test Set Lesion Cases:** 22 (100% of the test patients have $>0$ voxels in their mask)
- **Test Set Control Cases:** 0

**Conclusion on Controls:** There are **0 control (healthy) cases** in the provided preprocessed dataset. Every case has confirmed multiple sclerosis lesions as labeled in the masks. The clinical data (`info_dataset/clinical_data.csv`) also corroborates this, indicating that all subjects have a documented MS type (SMRR, SMSP, SMPP) with non-zero Lesion Volume and Lesion Number.

## 4. Suitability for U-Net Architecture

The preprocessed dataset (`MSLesSeg Dataset`) is **highly suitable** for a supervised 3D (or 2D) U-Net:
1. **Mask Availability:** Every single patient/timepoint in both `train` and `test` comes with a well-defined `_MASK.nii.gz`.
2. **Standardization:** All spatial dimensions are homogeneously `(182, 218, 182)`, which removes the need for complex resizing or resampling during data loading.
3. **Modalities:** FLAIR is completely matched in counts to the masks (93 in train, 22 in test), making it an excellent primary modality. T2 and T1 can be stacked as multi-channel inputs since they are perfectly registered to the same dimensions.
