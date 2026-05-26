# Preprocessing Walkthrough

This guide details how to compile multiple MR image datasets (ISBI2015, MICCAI, and MSLegSeg) into a standardized structure and apply advanced preprocessing techniques including Skull Stripping, Bias Correction, Z-Score Normalization, and Resizing.

## Pipeline Overview

The `preprocessing.py` script executes two major parts:

### Part I: Data Compilation
The script navigates through multiple raw data sources and compiles them into a unified format in the directory `dummy_model_2d/dataset/compiled/raw`. Each subject's data is anonymized and prefixed with a standardized `Si_` format (e.g., `S1_FLAIR.nii.gz` and `S1_MASK.nii.gz`).

Datasets processed:
1. **ISBI2015**: Extracts flair and mask files directly.
2. **MICCAI**: Extracts flair and ground truth files, actively transposing the axes from `(z, x, y)` to `(x, z, y)` as required.
3. **MSLegSeg (Train & Test)**: Syncs raw flair images from `MSLegSeg_RAW` with their corresponding masks from `MSLegSeg Dataset`.

### Part II: Preprocessing
All raw data in the `compiled/raw` folder automatically undergoes the following sequential steps, resulting in final files saved to `dummy_model_2d/dataset/compiled/preprocessed`:

1. **Skull Scraping (HD-BET)**: Uses the advanced deep-learning based HD-BET tool to isolate brain tissue from the skull and background.
2. **N4 Bias Field Correction (SimpleITK)**: Corrects low-frequency intensity non-uniformities caused by the magnetic field during MRI acquisition.
3. **Z-Score Normalization**: Standardizes voxel intensities specifically within the brain mask to have a mean of 0 and a standard deviation of 1.
4. **Resizing**: Resizes the `(x, y)` plane coordinates of all scans exactly to `256x256` using interpolation, preserving the `z` slices intact. Masks are resized using nearest-neighbor interpolation to maintain binary structures.

---

## Prerequisites & Installation

Before running the script, ensure you have the necessary libraries installed in your Python environment. 

### 1. Standard Python Packages
You need `numpy`, `nibabel`, `scikit-image`, and `SimpleITK`. Run:
```bash
pip install numpy nibabel scikit-image SimpleITK
```

### 2. HD-BET Installation
HD-BET (High-Definition Brain Extraction Tool) must be installed directly from its source repository to run from the command line:

```bash
# Clone the repository
git clone https://github.com/MIC-DKFZ/HD-BET
cd HD-BET

# Install the package
pip install -e .
```
*(Note: HD-BET will automatically download its pre-trained model weights the first time it is executed.)*

---

## Running the Pipeline

Once the requirements are met, you can execute the preprocessing pipeline by running:

```bash
cd /home/darshan/MS/dummy_model_2d
python3 preprocessing.py
```

### What to Expect:
- You will first see output indicating the compilation of subjects (e.g., `Compiling ISBI2015...`, `Compiling MICCAI...`).
- The script will state `Successfully compiled X subjects into .../compiled/raw`.
- Next, it will begin processing each subject. 
- You will see logs from HD-BET processing each image, followed by the script confirming the bias correction, normalization, and resizing steps.
- Processing large datasets can take a significant amount of time due to the computationally intensive nature of HD-BET and N4 Bias Correction.

Once finished, your fully preprocessed, model-ready datasets will be waiting inside `dummy_model_2d/dataset/compiled/preprocessed`.
