# 3-D U-Net Training Walkthrough

This walkthrough documents the $256 \times 256 \times 192$ multi-modal 3D U-Net pipeline implemented for the `unet3d` model. It explains the generator-based dataset loading, custom 3D loss functions, and how to execute training.

---

## Workspace Structure

The files inside `models/unet3d` are:
*   **[`src/model.py`](file:///home/darshan/MS/models/unet3d/src/model.py)**: The custom 3D U-Net model with lightweight layer widths to prevent VRAM OOM.
*   **[`src/dataset.py`](file:///home/darshan/MS/models/unet3d/src/dataset.py)**: Generator-based data loader. Symmetrically pads axial dimensions to $256 \times 256$ and the depth axis from $182$ to $192$ on-the-fly to construct shape `(256, 256, 192, 3)` in-memory.
*   **[`scripts/train.py`](file:///home/darshan/MS/models/unet3d/scripts/train.py)**: The 3D training script using a custom Keras 3D Dice Loss function.
*   **[`scripts/multi_runs.sh`](file:///home/darshan/MS/models/unet3d/scripts/multi_runs.sh)**: Parameter grid search shell script.
*   **[`results/test.ipynb`](file:///home/darshan/MS/models/unet3d/results/test.ipynb)**: Prediction visualizer notebook.

---

## Custom 3D Dice Loss

The `segmentation_models` library only supports 2D convolution tasks. To train the 3D U-Net, we implemented a custom 3D Dice loss inside `train.py`:
```python
def dice_loss_3d(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    axes = [1, 2, 3, 4]
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    union = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    return 1.0 - tf.reduce_mean(dice)
```
This is fully self-contained and avoids shape errors during 3D tensor training.

---

## How to Run Training

To start training the 3D U-Net on a single volume batch size (to prevent GPU memory exhaustions) on GPU `1` or `0`:

```bash
LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib \
/home/darshan/miniconda3/envs/venv/bin/python models/unet3d/scripts/train.py \
  --epochs 50 \
  --batch_size 1 \
  --lr 1e-5 \
  --optimizer adam \
  --lr_schedule fixed \
  --gpu 1
```

### Dry-run Verification Command
To run a fast validation check of the 3D pipeline:

```bash
LD_LIBRARY_PATH=/home/darshan/miniconda3/envs/venv/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/targets/x86_64-linux/lib \
/home/darshan/miniconda3/envs/venv/bin/python models/unet3d/scripts/train.py \
  --epochs 2 \
  --batch_size 1 \
  --gpu 1
```

### Multi-runs Grid Search
You can launch a background grid parameter search across all 6 configurations of learning rates and optimizers by running:
```bash
bash models/unet3d/scripts/multi_runs.sh
```
All outputs are directed to `models/unet3d/logs/`.
