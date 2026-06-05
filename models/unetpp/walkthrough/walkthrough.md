# 2-D UNet++ Training Walkthrough

This walkthrough documents the $256 \times 256$ nested multi-modal U-Net pipeline implemented for the `unetpp` model. It explains how to load, train, and evaluate the UNet++ model on-the-fly.

---

## Workspace Structure

The files inside `models/unetpp` are:
*   **[`src/model.py`](file:///home/darshan/MS/models/unetpp/src/model.py)**: The custom UNet++ architecture built in Keras.
*   **[`src/dataset.py`](file:///home/darshan/MS/models/unetpp/src/dataset.py)**: Loader utility that extracts slices directly from the preprocessed $256 \times 256$ dataset.
*   **[`scripts/train.py`](file:///home/darshan/MS/models/unetpp/scripts/train.py)**: Compiles and trains the model.
*   **[`results/test.ipynb`](file:///home/darshan/MS/models/unetpp/results/test.ipynb)**: Visualizes model predictions against MRI scans.

---

## How to Run Training

Run this command from the repository root to start the training pipeline on your selected GPU (e.g. GPU `1` or `0`):

```bash
LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib \
/home/darshan/miniconda3/envs/venv/bin/python models/unetpp/scripts/train.py \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-5 \
  --optimizer adam \
  --skip_blank_ratio 0.95 \
  --lr_schedule fixed \
  --gpu 1
```

### Dry-run Verification Command
To verify the full script execution:

```bash
LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib \
/home/darshan/miniconda3/envs/venv/bin/python models/unetpp/scripts/train.py \
  --epochs 2 \
  --batch_size 4 \
  --skip_blank_ratio 0.99 \
  --gpu 1
```

Inside `models/unetpp/runs/`, you will find your TensorBoard `logs/`, checkpoint `best_model.h5`, and `summary.json` containing patient-level validation/test confusion matrices.
