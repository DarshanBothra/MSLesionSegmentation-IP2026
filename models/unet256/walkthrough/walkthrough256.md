# 2-D U-Net (256x256) Training Walkthrough

This walkthrough documents the $256 \times 256$ multi-modal brain MRI lesion segmentation pipeline implemented for the `unet256` model. It explains the preprocessing dataset scaling, the architecture dimensions alignment, and how you can run the preprocessing and training pipelines.

---

## Directory Overview

The following files are located inside the `models/unet256` folder to keep the $256 \times 256$ pipeline completely separate from the original $182 \times 218$ version:

*   **[`preprocess256.py`](file:///home/darshan/MS/preprocessing/scripts/preprocess256.py)**: The preprocessing pipeline that outputs $256 \times 256 \times 182$ volumes.
*   **[`model256.py`](file:///home/darshan/MS/models/unet256/src/model256.py)**: The U-Net architecture compiled for $(256, 256, 3)$ inputs.
*   **[`dataset256.py`](file:///home/darshan/MS/models/unet256/src/dataset256.py)**: The data loader scanning `data/PREPROCESSED256` and serving $(256, 256, 3)$ slices.
*   **[`train256.py`](file:///home/darshan/MS/models/unet256/scripts/train256.py)**: The end-to-end training and evaluation loop.
*   **[`walkthrough256.md`](file:///home/darshan/MS/models/unet256/walkthrough/walkthrough256.md)**: This walkthrough document.

---

## Preprocessing & Image Resampling

To spatially normalize images to $256 \times 256$ in-plane resolution, the preprocessing pipeline implements the following stages in `preprocess256.py`:
1.  **Template Registration**: Align moving images to MNI152 template space (shape: $182 \times 218 \times 182$).
2.  **Brain Extraction**: FSL `bet` brain extraction.
3.  **N4 Bias Field Correction**: Inhomogeneity correction.
4.  **Z-score Intensity Normalization**: Inside the brain mask.
5.  **Final Resampling**: Resample each 3D volume using ANTsPy `ants.resample_image` to shape `(256, 256, 182)`. Modality scans use `linear` interpolation, and binary masks use `nearestNeighbor` (to preserve label integrity).

> [!NOTE]
> Resampling at the very end of preprocessing is the most accurate approach as it prevents bias correction and normalization parameters from being corrupted by interpolation artifacts.

---

## Architecture Dimensions Alignment

Unlike the original $182 \times 218$ shape, an input size of $256 \times 256$ is a perfect power of two ($2^8$). The downsampling division matches the upsampling multiplication cleanly:
*   **Input**: $256 \times 256$
*   **Level 1 Pooling**: $128 \times 128$
*   **Level 2 Pooling**: $64 \times 64$
*   **Level 3 Pooling**: $32 \times 32$
*   **Level 4 Pooling**: $16 \times 16$
*   **Level 5 (Bottleneck)**: $16 \times 16$

Upsampling levels upscale dimensions back up: $16 \to 32 \to 64 \to 128 \to 256$.
Because the shapes match exactly at every stage, the dynamic padding/cropping logic (`_match_and_pad`) does not execute any operations (but it remains in `model256.py` for fallback safety if you change input shapes in the future).

---

## How to Run Preprocessing

Run the following command from the repository root to start the $256 \times 256$ preprocessing script:

```bash
LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib \
/home/darshan/miniconda3/envs/venv/bin/python preprocessing/scripts/preprocess256.py
```

This will output all preprocessed files to `data/PREPROCESSED256/<subject_id>/`.

---

## How to Run Training

Run the following command from the repository root to train `unet256` on your selected GPU (e.g. GPU `1` or `0`):

```bash
LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib \
/home/darshan/miniconda3/envs/venv/bin/python models/unet256/scripts/train256.py \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-5 \
  --optimizer adam \
  --skip_blank_ratio 0.95 \
  --lr_schedule fixed \
  --gpu 1
```

### Dry-run Verification Command
To verify the full script logic (compilation, data loading, training iteration, validation, metric evaluation) using a quick 2-epoch pass with a high blank slice filter (99% discarded):

```bash
LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib \
/home/darshan/miniconda3/envs/venv/bin/python models/unet256/scripts/train256.py \
  --epochs 2 \
  --batch_size 4 \
  --skip_blank_ratio 0.99 \
  --gpu 1
```

Inside `models/unet256/runs/`, you will find your TensorBoard `logs/`, the best weights checkpoint `best_model.h5`, the `training_log.csv`, and validation/test confusion matrices saved inside `summary.json`.
