# ISBI2015 Dataset Exploratory Data Analysis (EDA)

The **ISBI2015 Longitudinal Multiple Sclerosis Lesion Segmentation Dataset** contains two primary splits of data within the workspace:
1. **Training Set (`training/`)**: Data from 5 subjects with multiple timepoints and ground-truth masks.
2. **Test Set (`testdata_website/`)**: Data from 14 subjects with multiple timepoints (without masks, intended for challenge submission).

Each subject folder is further divided into two subsets: `orig` (original/raw scans) and `preprocessed` (standardized/resampled scans).

---

## 1. ISBI2015 Training Set (`training/`)

This directory contains longitudinal scans for 5 patients, each with 4-5 timepoints, totaling 21 distinct studies.

### A. Original Scans (`orig/`)
- **Total Patients:** 5 (`training01` through `training05`)
- **Total Timepoints/Studies:** 21
- **Modalities Count:**
  - **MPRAGE (T1):** 21 images
  - **T2:** 21 images
  - **PD:** 21 images
  - **FLAIR:** 21 images
- **Image Dimensions:** Heterogeneous and not co-registered across all modalities:
  - T1 (MPRAGE): Typically `(256, 256, 120)`
  - T2 / PD: Typically `(256, 256, 70)`
  - FLAIR: Mixed between `(256, 256, 70)` and `(256, 256, 35)`

### B. Preprocessed Scans (`preprocessed/`)
The preprocessed folders are standardized, making them ready for deep learning models.
- **Image Dimensions:** All images perfectly resampled and registered to **`(181, 217, 181)`**.
- **Modalities Count:** 21 images per modality (MPRAGE, T2, PD, FLAIR).
- **Masks (`masks/`):** 42 masks in total. The dataset provides **two ground-truth masks per timepoint** (from two independent human raters). Every timepoint has its masks.

---

## 2. ISBI2015 Test Set (`testdata_website/`)

This directory contains longitudinal scans for 14 patients for the challenge evaluation phase.

### A. Original Scans (`orig/`)
- **Total Patients:** 14 (`test01` through `test14`)
- **Total Timepoints/Studies:** 61
- **Modalities Count:**
  - **MPRAGE (T1):** 61 images
  - **T2:** 61 images
  - **PD:** 61 images
  - **FLAIR:** 61 images
- **Image Dimensions:** Similar to the training set's `orig` directory, dimensions vary (e.g., `(256, 256, 120)` vs `(256, 256, 70)` vs `(256, 256, 35)`).

### B. Preprocessed Scans (`preprocessed/`)
- **Image Dimensions:** All images perfectly standardized and registered to **`(181, 217, 181)`**.
- **Modalities Count:** 61 images per modality.
- **Masks:** 0 masks (Ground truth is withheld for the challenge leaderboard).

---

## 3. Analysis of Controls vs. Lesion Cases

To evaluate whether there are control (healthy) cases in the training set, the `masks/` directory was parsed and verified for lesion labels (`> 0`).

- **Train Set Lesion Cases:** 21 (100% of the training timepoints contain MS lesions in their masks)
- **Train Set Control Cases:** 0 (No masks are completely empty)

**Conclusion:** Similar to the MSLesSeg dataset, the ISBI2015 training set does not contain any healthy control patients; all provided cases are confirmed MS patients with visible lesions.

## 4. Suitability for U-Net Architecture

The preprocessed subset of ISBI2015 (`preprocessed/`) is **highly suitable** for a 3D or 2D U-Net:
1. **Mask Availability:** The training set provides 2 expert segmentations per timepoint, allowing models to learn from a consensus mask or evaluate inter-rater variability.
2. **Standardization:** All spatial dimensions across all modalities (T1, T2, PD, FLAIR) are `(181, 217, 181)`, eliminating the need for dynamic resizing during data-loading. 
3. **Multi-Channel Input:** Because all 4 modalities are perfectly co-registered, they can easily be concatenated into a 4-channel input `[FLAIR, T1, T2, PD]` for the initial layer of a neural network.
