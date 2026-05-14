# 2D U-Net Model for MS Lesion Segmentation: Walkthrough & Training Guide

This guide details the complete workflow for training the 2D U-Net model with the newly requested architectural updates (Dropout & Batch Normalization in downsampling, Dropout in upsampling), running a comprehensive Hyperparameter Grid Search, and generating inference results.

## Prerequisites & Installation
Ensure you are running within your deep learning environment.

Required packages (assuming TensorFlow 2.x):
```bash
pip install tensorflow segmentation-models nibabel numpy pandas matplotlib
```

## Architectural Changes Implemented
The `model.py` and `train.py` have been entirely rewritten to strictly follow the requested architecture and training properties:
- **Downsampling**: Batch Normalization is added after *every* convolution. A Dropout layer is added after the *first* convolution of every block. The dropout rates are `0.1` (blocks 1, 2), `0.2` (blocks 3, 4), and `0.3` (block 5).
- **Upsampling**: A Dropout layer is added *before the last convolution* of every upsampling block. The dropout rates are `0.2` (blocks 6, 7), and `0.1` (blocks 8, 9).
- **Loss Function**: `sm.losses.dice_loss` is utilized.
- **Metrics**: `IoU` and `Dice Score` metrics are actively tracked.
- **Layer Output Info**: The scripts print the dimensions of layers going down the encoder and up the decoder for the very first sample.

## 1. Running the Grid Search (Cross Validation)
To determine the best combinations of Batch Size (4, 8, 16), Optimizers (Adam, SGD), and Learning Rates (1e-3, 1e-4, 1e-5), we use the newly created `grid_search.py` script.

The grid search will iterate over all 18 combinations. It uses early stopping based on the validation dice score with a maximum of 100 epochs.

Run the grid search script:
```bash
python grid_search.py
```
**What this does:**
1. Iterates through all 18 hyperparameter combinations.
2. For each combination, it invokes `train.py` which:
   - Prints intermediate layer shapes for the first sample.
   - Streams training live to TensorBoard.
   - Generates a `prediction_sample.png` of one sample from the test dataset (displaying T1w, T2w, FLAIR, Ground Truth, and Predicted Mask) at the end of each run in `runs/<timestamp>/`.
3. Tabulates the results of each combination into a console markdown table and stores them in `grid_search_results.csv`.

## 2. Using TensorBoard
To track live updates, errors, scores, and metrics vs. epochs visually, you can start TensorBoard. `train.py` is configured to output logs to the `logs/` directory inside each run.

Run TensorBoard in a separate terminal:
```bash
tensorboard --logdir runs/
```
You can access the dashboard by navigating to `http://localhost:6006` in your browser.

## 3. Running a Single Training Session Manually
If you want to train the model individually using the best hyperparameter configuration found during the grid search:

```bash
python train.py --epochs 100 --batch_size 8 --optimizer adam --lr 1e-4
```

## 4. Predicting and Visualizing the Middle Slice
As requested, a completely independent prediction script `predict_sample.py` is available. It processes raw `.nii.gz` sequences (T1w, T2w, FLAIR), generates the lesion mask, and produces a visualization of the **middle slice**. The output mask assigns white pixels to lesions and black to the background.

```bash
python predict_sample.py \
    --t1 /home/darshan/MS/model_dataset/test/P41_T1/t1.nii.gz \
    --t2 /home/darshan/MS/model_dataset/test/P41_T1/t2.nii.gz \
    --flair /home/darshan/MS/model_dataset/test/P41_T1/flair.nii.gz \
    --gt /home/darshan/MS/model_dataset/test/P41_T1/mask.nii.gz \
    --model /home/darshan/MS/dummy_model_2d/runs/<YOUR_BEST_RUN_ID>/best_model.h5 \
    --output_dir ./my_prediction_output
```

This will output:
- `predicted_mask.nii.gz`: The entire predicted 3D mask volume.
- `middle_slice_prediction.png`: A 5-panel figure displaying T1w, T2w, FLAIR, GT, and Predicted Mask for the middle slice.
