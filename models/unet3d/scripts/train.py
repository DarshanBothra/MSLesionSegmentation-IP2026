"""
train.py
--------
End-to-end training script for the 3-D U-Net on the MS preprocessed multimodal dataset.
Converts input volumes to 256x256x192 on the fly.
Modalities used: T1w + T2w + FLAIR (3-channel input)
"""

import os
import sys

# Pre-parse --gpu argument to set CUDA_VISIBLE_DEVICES before importing tensorflow
gpu_id = "0"
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        gpu_id = sys.argv[i + 1]
        break

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
os.environ["SM_FRAMEWORK"] = "tf.keras"

import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")          # Safe for headless environments
import matplotlib.pyplot as plt

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import Model3D
from dataset import build_datasets, load_and_preprocess_subject_3d

# ── GPU Configuration ─────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("--> GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(e)

# ── Custom 3D Keras metrics & Loss ───────────────────────────────────────────

class DiceScore3D(tf.keras.metrics.Metric):
    """Mean 3D Dice Similarity Coefficient (binary, threshold at 0.5)."""

    def __init__(self, threshold=0.5, name="dice_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.dice_sum   = self.add_weight(name="dice_sum",   initializer="zeros")
        self.batch_count = self.add_weight(name="batch_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true,                  tf.float32)

        # Compute per-sample 3D Dice (axes: H, W, D, C)
        axes   = [1, 2, 3, 4]
        inter  = tf.reduce_sum(y_true * y_pred, axis=axes)
        union  = tf.reduce_sum(y_true,          axis=axes) \
               + tf.reduce_sum(y_pred,          axis=axes)
        
        denom = union + 1e-7
        dice = (2.0 * inter + 1e-7) / denom
        
        self.dice_sum.assign_add(tf.reduce_mean(dice))
        self.batch_count.assign_add(1.0)

    def result(self):
        return self.dice_sum / (self.batch_count + 1e-7)

    def reset_state(self):
        self.dice_sum.assign(0.0)
        self.batch_count.assign(0.0)


class IoUScore3D(tf.keras.metrics.Metric):
    """Mean 3D Intersection-over-Union (binary, threshold at 0.5)."""

    def __init__(self, threshold=0.5, name="iou_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold  = threshold
        self.iou_sum    = self.add_weight(name="iou_sum",    initializer="zeros")
        self.batch_count = self.add_weight(name="batch_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred  = tf.cast(y_pred > self.threshold, tf.float32)
        y_true  = tf.cast(y_true,                  tf.float32)

        axes   = [1, 2, 3, 4]
        inter  = tf.reduce_sum(y_true * y_pred, axis=axes)
        union  = tf.reduce_sum(y_true,          axis=axes) \
               + tf.reduce_sum(y_pred,          axis=axes) \
               - inter
        
        iou    = (inter + 1e-7) / (union + 1e-7)
        self.iou_sum.assign_add(tf.reduce_mean(iou))
        self.batch_count.assign_add(1.0)

    def result(self):
        return self.iou_sum / (self.batch_count + 1e-7)

    def reset_state(self):
        self.iou_sum.assign(0.0)
        self.batch_count.assign(0.0)


def dice_loss_3d(y_true, y_pred):
    """Custom 3-D Dice loss function for Keras."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    axes = [1, 2, 3, 4]
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    union = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
    return 1.0 - tf.reduce_mean(dice)

# ── Dynamic Run Directory Resolution ──────────────────────────────────────────

def resolve_run_dir(runs_dir: str, optimizer_name: str, lr: float, batch_size: int) -> tuple:
    """
    Dynamically resolve the run directory name exactly as:
    <optimizer_name>_<lr>_<batch_size>_<iteration_number_for_combination>
    """
    prefix = f"{optimizer_name.lower()}_{lr}_{batch_size}_"
    os.makedirs(runs_dir, exist_ok=True)
    existing_runs = [d for d in os.listdir(runs_dir) if d.startswith(prefix) and os.path.isdir(os.path.join(runs_dir, d))]
    
    max_iteration = 0
    for run_name in existing_runs:
        suffix = run_name[len(prefix):]
        try:
            val = int(suffix)
            if val > max_iteration:
                max_iteration = val
        except ValueError:
            pass
            
    iteration = max_iteration + 1
    run_id = f"{prefix}{iteration}"
    return run_id, os.path.join(runs_dir, run_id)

# ── Model Compilation ─────────────────────────────────────────────────────────

def build_model(lr: float = 1e-5, optimizer_name: str = "adam") -> tf.keras.Model:
    """Instantiate Model3D and compile it with 3D Dice loss, DiceScore3D, IoUScore3D."""
    m3d = Model3D(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_DEPTH=192, IMG_CHANNELS=3)
    m3d.initializeModel()

    if optimizer_name.lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    m3d.model.compile(
        optimizer=optimizer,
        loss=dice_loss_3d,
        metrics=[
            "accuracy",
            DiceScore3D(name="dice_score"),
            IoUScore3D(name="iou_score"),
        ],
    )
    return m3d.model

# ── Plotting Helpers ─────────────────────────────────────────────────────────

def plot_training_curves(history, run_dir: str):
    """Save Dice and IoU metrics plots over epochs."""
    epochs = range(1, len(history.history["loss"]) + 1)

    def _save(metric_key, val_metric_key, ylabel, title, filename):
        train_vals = history.history.get(metric_key, [])
        test_vals  = history.history.get(val_metric_key, [])

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, train_vals, "b-o",  markersize=4, linewidth=1.5, label="Train")
        if test_vals:
            ax.plot(epochs, test_vals,  "r--s", markersize=4, linewidth=1.5, label="Val")

            best_epoch = int(np.argmax(test_vals)) + 1 if "score" in metric_key else int(np.argmin(test_vals)) + 1
            ax.axvline(best_epoch, color="grey", linestyle=":", alpha=0.7,
                       label=f"Best epoch = {best_epoch}")
            ax.scatter([best_epoch], [test_vals[best_epoch - 1]],
                       color="red", zorder=5, s=60)

        ax.set_xlabel("Epochs", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1, max(epochs)])
        if "score" in metric_key:
            ax.set_ylim([0, 1])

        out_path = os.path.join(run_dir, filename)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved plot: {out_path}")

    _save("dice_score", "val_dice_score",
          "3D Dice Coefficient",
          "3-D U-Net — Dice Coefficient vs Epochs (T1w + T2w + FLAIR)",
          "dice_curve.png")

    _save("iou_score", "val_iou_score",
          "3D IoU Score",
          "3-D U-Net — IoU Score vs Epochs (T1w + T2w + FLAIR)",
          "iou_curve.png")

# ── Confusion Matrix Evaluation ──────────────────────────────────────────────

def evaluate_confusion_matrix(model, split_name, subjects, preprocessed_dir):
    """Dynamically evaluate patient-level confusion matrix in 3D."""
    print(f"\n{'='*60}")
    print(f"COMPUTING 3D CONFUSION MATRIX FOR {split_name.upper()} SPLIT (PER PATIENT)")
    print(f"{'='*60}")
    
    header = f"{'Subject':<12} | {'TP':>12} | {'TN':>12} | {'FP':>12} | {'FN':>12}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    
    for subject_id in subjects:
        subj_dir = os.path.join(preprocessed_dir, subject_id)
        
        try:
            img, mask = load_and_preprocess_subject_3d(subj_dir, target_h=256, target_w=256, target_d=192)
            if img is None:
                continue
                
            # Predict (shape: 1, 256, 256, 192, 3)
            pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
            pred_bin = (pred[0] > 0.5).astype(np.float32)
            
            tp = int(np.sum((mask == 1.0) & (pred_bin == 1.0)))
            tn = int(np.sum((mask == 0.0) & (pred_bin == 0.0)))
            fp = int(np.sum((mask == 0.0) & (pred_bin == 1.0)))
            fn = int(np.sum((mask == 1.0) & (pred_bin == 0.0)))
            
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            
            print(f"{subject_id:<12} | {tp:>12,} | {tn:>12,} | {fp:>12,} | {fn:>12,}")
        except Exception as e:
            print(f"  [WARN] Confusion matrix evaluation failed for {subject_id}: {e}")
            
    print(sep)
    print(f"{'TOTAL':<12} | {total_tp:>12,} | {total_tn:>12,} | {total_fp:>12,} | {total_fn:>12,}")
    
    dice_3d = float((2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + 1e-7))
    iou_3d = float(total_tp / (total_tp + total_fp + total_fn + 1e-7))
    
    print(f"{'METRICS':<12} | 3D Dice: {dice_3d:.4f} | 3D IoU: {iou_3d:.4f}")
    print(sep + "\n")
    
    return {
        "TP": total_tp,
        "TN": total_tn,
        "FP": total_fp,
        "FN": total_fn,
        "dice_3d": round(dice_3d, 4),
        "iou_3d": round(iou_3d, 4)
    }

# ── Visualization Sample ─────────────────────────────────────────────────────

def save_prediction_sample(model, test_ds, run_dir):
    """Save a sample prediction visualization image (center slice of the 3D volume)."""
    print("\nSaving prediction sample visual...")
    for x, y in test_ds.take(1):
        pred = model.predict(x, verbose=0)
        
        # Take center slice along the depth axis (z = 96)
        z = 96
        
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        
        t1 = x[0, :, :, z, 0].numpy()
        t2 = x[0, :, :, z, 1].numpy()
        flair = x[0, :, :, z, 2].numpy()
        gt = y[0, :, :, z, 0].numpy()
        p = pred[0, :, :, z, 0]
        
        axes[0].imshow(t1, cmap='gray')
        axes[0].set_title(f'T1w (z={z})')
        axes[0].axis('off')
        
        axes[1].imshow(t2, cmap='gray')
        axes[1].set_title(f'T2w (z={z})')
        axes[1].axis('off')
        
        axes[2].imshow(flair, cmap='gray')
        axes[2].set_title(f'FLAIR (z={z})')
        axes[2].axis('off')
        
        axes[3].imshow(gt, cmap='gray')
        axes[3].set_title('Ground Truth')
        axes[3].axis('off')
        
        axes[4].imshow(p > 0.5, cmap='gray')
        axes[4].set_title('Predicted Mask')
        axes[4].axis('off')
        
        plt.tight_layout()
        pred_path = os.path.join(run_dir, "prediction_sample.png")
        plt.savefig(pred_path, dpi=150)
        plt.close(fig)
        print(f"  Saved prediction sample visual: {pred_path}")
        break

# ── Main Training Loop ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train 3-D U-Net for multi-modal brain lesion segmentation")
    parser.add_argument("--epochs",      type=int,   default=50,     help="Max training epochs")
    parser.add_argument("--batch_size",  type=int,   default=1,      help="Batch size (highly recommended to use 1)")
    parser.add_argument("--lr",          type=float, default=1e-5,   help="Learning rate")
    parser.add_argument("--optimizer",   type=str,   default="adam", help="Optimizer (adam, sgd)")
    parser.add_argument(
        "--lr_schedule",
        type=str,
        choices=["fixed", "plateau"],
        default="fixed",
        help="Learning rate schedule: 'fixed' or 'plateau'")
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU ID to use (e.g. '0' or '1')"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "data", "PREPROCESSED"))
    runs_dir = os.path.abspath(os.path.join(script_dir, "..", "runs"))

    # 1. Resolve Dynamic Run Directory Naming
    run_id, run_dir = resolve_run_dir(runs_dir, args.optimizer, args.lr, args.batch_size)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nRun ID: {run_id}")
    print(f"Run directory: {run_dir}")

    # 2. Build Datasets
    train_ds, val_ds, test_ds, split_raw = build_datasets(
        preprocessed_dir=preprocessed_dir,
        batch_size=args.batch_size
    )

    # 3. Build & Compile Model
    model = build_model(lr=args.lr, optimizer_name=args.optimizer)
    try:
        model.summary(line_length=100)
    except Exception as e:
        print(f"Could not print model summary: {e}")

    # 4. Define Callbacks
    ckpt_path = os.path.join(run_dir, "best_model.h5")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_dice_score",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(run_dir, "training_log.csv")
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(run_dir, "logs"),
            histogram_freq=1
        )
    ]

    if args.lr_schedule == "plateau":
        print(f"  LR schedule: ReduceLROnPlateau (factor=0.5, patience=5, min_lr=1e-7)")
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_dice_score",
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        )
    else:
        print(f"  LR schedule: fixed (lr={args.lr})")

    # 5. Model Fit
    print(f"\nStarting training for up to {args.epochs} epochs ...\n")
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Save Plot Training Curves
    print("\nSaving training curves...")
    plot_training_curves(history, run_dir)

    # 7. Evaluate on Held-Out Test Set
    print("\n" + "="*60)
    print("EVALUATING MODEL ON HELD-OUT TEST SET")
    print("="*60)
    test_results = model.evaluate(test_ds, verbose=1)
    test_metrics = dict(zip(model.metrics_names, test_results))

    # 9. Confusion Matrix on test/val splits
    test_cm = evaluate_confusion_matrix(model, "Test", split_raw["test"], preprocessed_dir)
    val_cm = evaluate_confusion_matrix(model, "Validation", split_raw["val"], preprocessed_dir)

    # 10. Save Run Summary JSON
    best_val_dice = max(history.history.get("val_dice_score", [0]))
    best_val_iou  = max(history.history.get("val_iou_score",  [0]))

    summary = {
        "run_id":           run_id,
        "lr":               args.lr,
        "optimizer":        args.optimizer,
        "lr_schedule":      args.lr_schedule,
        "epochs_trained":   len(history.history["loss"]),
        "val_split": {
            "best_dice": round(float(best_val_dice), 4),
            "best_iou":  round(float(best_val_iou),  4),
        },
        "test_split": {
            k: round(float(v), 4) for k, v in test_metrics.items()
        },
        "validation_confusion_matrix": val_cm,
        "test_confusion_matrix":       test_cm,
        "checkpoint":       ckpt_path,
        "training_log":     os.path.join(run_dir, "training_log.csv")
    }

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING & EVALUATION COMPLETE")
    print("="*60)
    print(f"  Best Val Dice: {best_val_dice:.4f}")
    print(f"  Best Val IoU:  {best_val_iou:.4f}")
    print(f"  Test Dice:     {test_metrics.get('dice_score', 'N/A')}")
    print(f"  Test IoU:      {test_metrics.get('iou_score', 'N/A')}")
    print(f"\nSummary JSON saved to: {summary_path}")

if __name__ == "__main__":
    main()
