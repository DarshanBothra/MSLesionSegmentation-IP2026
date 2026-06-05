"""
train.py
--------
End-to-end training script for the 2-D UNet++ on the MS preprocessed multimodal dataset.
Converts input slices to 256x256 on the fly.
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
import segmentation_models as sm
import matplotlib
matplotlib.use("Agg")          # Safe for headless environments
import matplotlib.pyplot as plt

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import UNetPlusPlus
from dataset import build_datasets, extract_slices

# ── GPU Configuration ─────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("--> GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(e)

# ── Custom Keras metrics ─────────────────────────────────────────────────────

class DiceScore(tf.keras.metrics.Metric):
    """Mean Dice Similarity Coefficient (binary, threshold at 0.5)."""

    def __init__(self, threshold=0.5, name="dice_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.dice_sum   = self.add_weight(name="dice_sum",   initializer="zeros")
        self.batch_count = self.add_weight(name="batch_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true,                  tf.float32)

        # Compute per-sample Dice
        axes   = [1, 2, 3]                    # H, W, C
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


class IoUScore(tf.keras.metrics.Metric):
    """Mean Intersection-over-Union (binary, threshold at 0.5)."""

    def __init__(self, threshold=0.5, name="iou_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold  = threshold
        self.iou_sum    = self.add_weight(name="iou_sum",    initializer="zeros")
        self.batch_count = self.add_weight(name="batch_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred  = tf.cast(y_pred > self.threshold, tf.float32)
        y_true  = tf.cast(y_true,                  tf.float32)

        axes   = [1, 2, 3]
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
    """Instantiate UNetPlusPlus and compile it with Dice loss, DiceScore, IoUScore."""
    m2d = UNetPlusPlus(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3)
    m2d.initializeModel()

    if optimizer_name.lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name.lower() == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    m2d.model.compile(
        optimizer=optimizer,
        loss=sm.losses.dice_loss,
        metrics=[
            "accuracy",
            DiceScore(name="dice_score"),
            IoUScore(name="iou_score"),
        ],
    )
    return m2d.model

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
          "Dice Coefficient",
          "2-D UNet++ — Dice Coefficient vs Epochs (T1w + T2w + FLAIR)",
          "dice_curve.png")

    _save("iou_score", "val_iou_score",
          "IoU Score",
          "2-D UNet++ — IoU Score vs Epochs (T1w + T2w + FLAIR)",
          "iou_curve.png")

# ── Confusion Matrix Evaluation ──────────────────────────────────────────────

def evaluate_confusion_matrix(model, split_name, subjects, preprocessed_dir, batch_size):
    """Dynamically evaluate patient-level confusion matrix."""
    print(f"\n{'='*60}")
    print(f"COMPUTING CONFUSION MATRIX FOR {split_name.upper()} SPLIT (PER PATIENT)")
    print(f"{'='*60}")
    
    header = f"{'Subject':<12} | {'TP':>12} | {'TN':>12} | {'FP':>12} | {'FN':>12} | {'Dice (3D)':>10} | {'IoU (3D)':>10}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0
    
    dice_scores = []
    iou_scores = []
    subject_details = []
    
    for subject_id in subjects:
        subj_dir = os.path.join(preprocessed_dir, subject_id)
        t1_path = os.path.join(subj_dir, "t1.nii.gz")
        t2_path = os.path.join(subj_dir, "t2.nii.gz")
        flair_path = os.path.join(subj_dir, "flair.nii.gz")
        mask_path = os.path.join(subj_dir, "mask.nii.gz")
        
        if not (os.path.exists(t1_path) and os.path.exists(t2_path) and os.path.exists(flair_path) and os.path.exists(mask_path)):
            continue
            
        try:
            imgs, masks = extract_slices(t1_path, t2_path, flair_path, mask_path,
                                         skip_blank_ratio=0.0, is_train=False)
            if imgs is None:
                continue
                
            preds = model.predict(imgs, batch_size=batch_size, verbose=0)
            preds_bin = (preds > 0.5).astype(np.float32)
            
            tp = int(np.sum((masks == 1.0) & (preds_bin == 1.0)))
            tn = int(np.sum((masks == 0.0) & (preds_bin == 0.0)))
            fp = int(np.sum((masks == 0.0) & (preds_bin == 1.0)))
            fn = int(np.sum((masks == 1.0) & (preds_bin == 0.0)))
            
            # Compute 3D Volume Dice & IoU for this subject
            denom_dice = 2.0 * tp + fp + fn
            dice = 1.0 if denom_dice == 0 else (2.0 * tp) / denom_dice
            
            denom_iou = tp + fp + fn
            iou = 1.0 if denom_iou == 0 else float(tp) / denom_iou
            
            dice_scores.append(dice)
            iou_scores.append(iou)
            
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            
            print(f"{subject_id:<12} | {tp:>12,} | {tn:>12,} | {fp:>12,} | {fn:>12,} | {dice:>10.4f} | {iou:>10.4f}")
            subject_details.append({
                "subject_id": subject_id,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "dice_3d": round(dice, 4),
                "iou_3d": round(iou, 4)
            })
        except Exception as e:
            print(f"  [WARN] Confusion matrix evaluation failed for {subject_id}: {e}")
            
    print(sep)
    # Compute split-level metrics
    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
    mean_iou = np.mean(iou_scores) if iou_scores else 0.0
    
    # Global/pooled volume Dice and IoU
    pooled_denom_dice = 2.0 * total_tp + total_fp + total_fn
    pooled_dice = 1.0 if pooled_denom_dice == 0 else (2.0 * total_tp) / pooled_denom_dice
    
    pooled_denom_iou = total_tp + total_fp + total_fn
    pooled_iou = 1.0 if pooled_denom_iou == 0 else float(total_tp) / pooled_denom_iou

    print(f"{'TOTAL':<12} | {total_tp:>12,} | {total_tn:>12,} | {total_fp:>12,} | {total_fn:>12,} | {pooled_dice:>10.4f} | {pooled_iou:>10.4f}")
    print(f"{'MEAN (Avg)':<12} | {'-':>12} | {'-':>12} | {'-':>12} | {'-':>12} | {mean_dice:>10.4f} | {mean_iou:>10.4f}")
    print(sep + "\n")
    
    return {
        "TP": total_tp,
        "TN": total_tn,
        "FP": total_fp,
        "FN": total_fn,
        "mean_3d_dice": round(mean_dice, 4),
        "mean_3d_iou": round(mean_iou, 4),
        "pooled_3d_dice": round(pooled_dice, 4),
        "pooled_3d_iou": round(pooled_iou, 4),
        "subjects": subject_details
    }

# ── Visualization Sample ─────────────────────────────────────────────────────

def save_prediction_sample(model, test_ds, run_dir):
    """Save a sample prediction visualization image containing T1, T2, FLAIR, GT, and Pred."""
    print("\nSaving prediction sample visual...")
    for x, y in test_ds.take(1):
        sample_idx = 0
        for i in range(x.shape[0]):
            if tf.reduce_sum(y[i]).numpy() > 0:
                sample_idx = i
                break
                
        pred = model.predict(x[sample_idx:sample_idx+1], verbose=0)
        
        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        
        t1 = x[sample_idx, :, :, 0].numpy()
        t2 = x[sample_idx, :, :, 1].numpy()
        flair = x[sample_idx, :, :, 2].numpy()
        gt = y[sample_idx, :, :, 0].numpy()
        p = pred[0, :, :, 0]
        
        axes[0].imshow(t1, cmap='gray')
        axes[0].set_title('T1w')
        axes[0].axis('off')
        
        axes[1].imshow(t2, cmap='gray')
        axes[1].set_title('T2w')
        axes[1].axis('off')
        
        axes[2].imshow(flair, cmap='gray')
        axes[2].set_title('FLAIR')
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
    parser = argparse.ArgumentParser(description="Train 2-D UNet++ for multi-modal brain lesion segmentation")
    parser.add_argument("--epochs",      type=int,   default=50,     help="Max training epochs")
    parser.add_argument("--batch_size",  type=int,   default=4,      help="Batch size")
    parser.add_argument("--lr",          type=float, default=1e-5,   help="Learning rate")
    parser.add_argument("--optimizer",   type=str,   default="adam", help="Optimizer (adam, sgd)")
    parser.add_argument(
        "--lr_schedule",
        type=str,
        choices=["fixed", "plateau"],
        default="fixed",
        help="Learning rate schedule: 'fixed' or 'plateau'")
    parser.add_argument(
        "--skip_blank_ratio",
        type=float,
        default=0.95,
        help="Fraction of lesion-free training slices to drop (0.0 to keep all)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU ID to use (e.g. '0' or '1')"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    preprocessed_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "data", "PREPROCESSED256"))
    runs_dir = os.path.abspath(os.path.join(script_dir, "..", "runs"))

    # 1. Resolve Dynamic Run Directory Naming
    run_id, run_dir = resolve_run_dir(runs_dir, args.optimizer, args.lr, args.batch_size)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nRun ID: {run_id}")
    print(f"Run directory: {run_dir}")

    # 2. Build Datasets
    train_ds, val_ds, test_ds, split_raw = build_datasets(
        preprocessed_dir=preprocessed_dir,
        batch_size=args.batch_size,
        skip_blank_ratio=args.skip_blank_ratio
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
    
    # Load best model weights for post-training evaluation
    if os.path.exists(ckpt_path):
        print(f"\nLoading best model weights from {ckpt_path} for final evaluation...")
        model.load_weights(ckpt_path)
    else:
        print(f"\n[WARN] Best model checkpoint not found at {ckpt_path}. Evaluating with final weights.")

    test_results = model.evaluate(test_ds, verbose=1)
    test_metrics = dict(zip(model.metrics_names, test_results))

    # 9. Confusion Matrix on test/val splits
    test_cm = evaluate_confusion_matrix(model, "Test", split_raw["test"][2], preprocessed_dir, args.batch_size)
    val_cm = evaluate_confusion_matrix(model, "Validation", split_raw["val"][2], preprocessed_dir, args.batch_size)

    # 10. Save Run Summary JSON
    best_val_dice = max(history.history.get("val_dice_score", [0]))
    best_val_iou  = max(history.history.get("val_iou_score",  [0]))

    summary = {
        "run_id":           run_id,
        "lr":               args.lr,
        "optimizer":        args.optimizer,
        "lr_schedule":      args.lr_schedule,
        "skip_blank_ratio": args.skip_blank_ratio,
        "epochs_trained":   len(history.history["loss"]),
        "val_split": {
            "best_dice_slice_avg": round(float(best_val_dice), 4),
            "best_iou_slice_avg":  round(float(best_val_iou),  4),
            "mean_3d_dice":        val_cm.get("mean_3d_dice", 0.0),
            "mean_3d_iou":         val_cm.get("mean_3d_iou", 0.0),
            "pooled_3d_dice":      val_cm.get("pooled_3d_dice", 0.0),
            "pooled_3d_iou":       val_cm.get("pooled_3d_iou", 0.0),
        },
        "test_split": {
            "slice_dice": round(float(test_metrics.get("dice_score", 0.0)), 4) if "dice_score" in test_metrics else None,
            "slice_iou":  round(float(test_metrics.get("iou_score", 0.0)), 4) if "iou_score" in test_metrics else None,
            "mean_3d_dice": test_cm.get("mean_3d_dice", 0.0),
            "mean_3d_iou":  test_cm.get("mean_3d_iou", 0.0),
            "pooled_3d_dice": test_cm.get("pooled_3d_dice", 0.0),
            "pooled_3d_iou":  test_cm.get("pooled_3d_iou", 0.0),
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
    print(f"  Best Val Dice (2D Slice-Avg): {best_val_dice:.4f}")
    print(f"  Best Val IoU (2D Slice-Avg):  {best_val_iou:.4f}")
    print(f"  Test Dice (2D Slice-Avg):     {test_metrics.get('dice_score', 'N/A')}")
    print(f"  Test IoU (2D Slice-Avg):      {test_metrics.get('iou_score', 'N/A')}")
    print(f"  Validation Mean 3D Dice:       {val_cm.get('mean_3d_dice', 0.0):.4f}")
    print(f"  Validation Mean 3D IoU:        {val_cm.get('mean_3d_iou', 0.0):.4f}")
    print(f"  Test Mean 3D Dice:             {test_cm.get('mean_3d_dice', 0.0):.4f}")
    print(f"  Test Mean 3D IoU:              {test_cm.get('mean_3d_iou', 0.0):.4f}")
    print(f"\nSummary JSON saved to: {summary_path}")

if __name__ == "__main__":
    main()
