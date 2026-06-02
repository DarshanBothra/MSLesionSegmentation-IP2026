# 2-D U-Net Training Walkthrough (Multi-modal MRI Segmentation)

This walkthrough documents the multi-modal brain MRI lesion segmentation pipeline implemented for the 2-D U-Net. It details how the dataset is structured, how the training script works, how directories are dynamically generated, and how you can run the program.

---

## Directory Overview

We have introduced the following files to support robust, reproducible training:

*   **[`dataset.py`](file:///home/darshan/MS/models/unet/src/dataset.py)**: The TensorFlow data pipeline.
    *   Scans `data/PREPROCESSED/` for all 187 subjects.
    *   Splits subjects deterministically into **Train (80% / 149 subjs)**, **Validation (10% / 18 subjs)**, and **Test (10% / 20 subjs)** splits using a seed of `42` to guarantee no patient-level overlap (preventing data leakage).
    *   Loads all 4 NIfTI files per subject: `t1.nii.gz`, `t2.nii.gz`, `flair.nii.gz`, and `mask.nii.gz`.
    *   Min-Max normalizes each MRI modality volume individually to `[0, 1]` to align intensity ranges.
    *   Resizes each slice symmetrically via center-cropping or padding to exactly `(182, 218)` to match the U-Net spatial dimensions.
    *   Stacks T1, T2, and FLAIR slices together to create a 3-channel input slice of shape `(182, 218, 3)`.
    *   Discards a configurable percentage (default `95%`) of background-only (blank) slices from the training split to focus learning on actual lesions and drastically speed up training.
*   **[`train.py`](file:///home/darshan/MS/models/unet/scripts/train.py)**: The main end-to-end training and evaluation script.
    *   Compiles `Model2D` using Dice Loss (`segmentation_models.losses.dice_loss`) and custom binary Dice and IoU evaluation metrics.
    *   Supports dynamic learning rate scaling (constant or `ReduceLROnPlateau`).
    *   Logs performance metrics to a CSV history file and TensorBoard directory.
    *   Resolves run directories dynamically exactly to your requested naming specification.
    *   Executes full patient-level metrics and confusion matrix evaluations on validation and test splits upon training completion.
    *   Saves a `summary.json` containing configuration details and final results. (No static PNG visual is saved during training).

---

## Naming Specification for Runs & Logs

Both training and validation logs are saved dynamically inside `models/unet/runs/`.
Per your specification, the run directory name is formatted exactly as:

$$\text{\{optimizer\_name\}}\_\text{\{lr\}}\_\text{\{batch\_size\}}\_\text{\{iteration\_number\_for\_combination\}}$$

For example, when running with optimizer `adam`, learning rate `1e-5`, and batch size `4` for the first time:
```
models/unet/runs/adam_1e-05_4_1/
```
Running it again with the same configuration will automatically detect the existing folder and increment the iteration suffix:
```
models/unet/runs/adam_1e-05_4_2/
```

Inside this directory, the following logs and outputs are automatically generated:
*   `logs/`: TensorBoard event logs (for both training and validation metrics).
*   `best_model.h5`: The best weights checkpoint based on validation Dice Similarity Score.
*   `training_log.csv`: The CSV containing loss, accuracy, dice, and IoU values for each epoch.
*   `dice_curve.png` / `iou_curve.png`: Plots tracking training and validation curves.
*   `summary.json`: Detailed JSON containing all script parameters, best validation epoch results, test split evaluation metrics, and complete patient-level confusion matrices for both validation and test subsets.

---

## Interactive 3D Predictor & Slice Visualizer

To visualize model predictions dynamically for any patient in your test dataset alongside their original MRI files, we have created an interactive Jupyter notebook at:
*   **[`test.ipynb`](file:///home/darshan/MS/models/unet/results/test.ipynb)**

### Features of the Notebook:
1.  **3D prediction volume creation:** Stacks and predicts slices for the selected subject, center-crops/pads predictions back to their exact original spatial dimensions, and saves them as a standard NIfTI volume:
    `models/unet/results/<optimizer>_<lr>_<batch_size>_<iteration>/pred_mask.nii.gz`
2.  **Metrics computation:** Calculates patient-level Dice, IoU, and complete confusion matrix (TP, TN, FP, FN) and writes them to:
    `models/unet/results/<optimizer>_<lr>_<batch_size>_<iteration>/prediction_details.json`
3.  **Single-slider slice interactive viewer:** Uses a single dynamic slider to scroll through the entire 3D volume, showing T1, T2, FLAIR, Ground Truth, and Predicted Mask perfectly aligned in axial view.

### How to Run the Notebook:
1.  Launch your Jupyter server from the environment:
    ```bash
    LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib jupyter notebook
    ```
2.  Open `models/unet/results/test.ipynb`.
3.  Specify the `model_path` (pointing to your `.h5` checkpoint) and the `subject_id` (a test patient ID) in the Configuration cell.
4.  Run all cells to execute predictions, save the NIfTI + JSON results, and interactively visualize the aligned modal slices.

---

## How to Run the Program

Because your local conda environment (`venv`) requires specific library linking for Python packages like `matplotlib`, you must execute the script with `LD_LIBRARY_PATH` prepended to dynamically link conda environment libraries (resolving `libstdc++.so.6 CXXABI_1.3.15` mismatches) and locate the dynamic CUDA drivers, cuDNN, and runtime files:

### 1. Training Command (Standard Configuration)

Run this command from the `/home/darshan/MS` repository root directory to start the training pipeline on your selected GPU (e.g. GPU `1` or GPU `0`):

```bash
LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib \
/home/darshan/miniconda3/envs/venv/bin/python models/unet/scripts/train.py \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-5 \
  --optimizer adam \
  --skip_blank_ratio 0.95 \
  --lr_schedule fixed \
  --gpu 1
```

### 2. Supported Command-Line Arguments

You can easily adjust hyperparameters and specify the GPU device using the following command-line flags:

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--epochs` | `int` | `50` | Maximum number of training epochs. |
| `--batch_size` | `int` | `4` | Training and validation batch size. |
| `--lr` | `float` | `1e-5` | Learning rate for the model. |
| `--optimizer` | `str` | `adam` | Optimization algorithm: choose either `adam` or `sgd`. |
| `--lr_schedule` | `str` | `fixed` | Learning rate scheduling strategy: choose `fixed` (constant rate) or `plateau` (ReduceLROnPlateau). |
| `--skip_blank_ratio` | `float` | `0.95` | Fraction of background-only (blank) slices to discard from the training split (e.g. `0.95` drops 95%). Set to `0.0` to retain all slices. |
| `--gpu` | `str` | `0` | GPU device ID to use (e.g. `0` to run on GPU 0, or `1` to run on GPU 1). |

### 3. Verification & Dry-run Command

To perform a quick dry-run verification of the entire script logic (data loader, compilation, metrics, evaluation, visualization, logs saving) without waiting for a full training cycle, you can run a 2-epoch pass with a high blank slice filter (99% discarded) to keep the dataset size minimal on your selected GPU:

```bash
LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib \
/home/darshan/miniconda3/envs/venv/bin/python models/unet/scripts/train.py \
  --epochs 2 \
  --batch_size 4 \
  --skip_blank_ratio 0.99 \
  --gpu 1
```

---

## How to View Logs via TensorBoard

To view and inspect training progress, loss trends, Dice scores, and IoU values dynamically, launch TensorBoard from your terminal:

```bash
# Point TensorBoard to the UNet runs folder
tensorboard --logdir=models/unet/runs/
```

Then open your browser and navigate to the address returned (usually `http://localhost:6006`).
This displays charts for:
*   `loss` vs. `val_loss`
*   `dice_score` vs. `val_dice_score`
*   `iou_score` vs. `val_iou_score`
*   `accuracy` vs. `val_accuracy`
