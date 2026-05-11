"""
train.py
--------
End-to-end training script for the 2-D U-Net on the MSLesSeg dataset.

Modality used
-------------
  Multi-modal: FLAIR + T1w + T2w  (3-channel input: 256 × 256 × 3)
  Missing modalities filled with zeros automatically.

Splits & roles
--------------
  train/        : 90% of MSLesSeg patients – model training
  external_val/ : ISBI2015 5 subjects      – validation during model.fit() (monitors generalisation)
  test/         : 10% of MSLesSeg patients – completely held-out; evaluated ONCE after training

Blank-slice filtering (--skip_blank_ratio)
------------------------------------------
  0.95 (default) : drop 95% of background-only slices in the training set
  0.0            : keep all slices (no filtering)
  Applied only to the training split; val/test always use all slices.

Metrics reported every epoch
------------------------------
  - Binary Cross-Entropy  (loss)
  - Pixel Accuracy
  - Dice Similarity Score  (train & test)
  - IoU Score              (train & test)

Outputs per run (saved to runs/<run_id>/)
-----------------------------------------
  best_model.keras       – best checkpoint
  training_log.csv       – full epoch-by-epoch metrics
  summary.json           – final aggregate results
  dice_curve.png         – Dice Coefficient vs Epochs (train + test)
  iou_curve.png          – IoU Score vs Epochs (train + test)

Usage
-----
  # Fixed learning rate (default, 1e-5):
  python train.py [--epochs 50] [--batch_size 8] [--lr 1e-5]

  # With ReduceLROnPlateau scheduler:
  python train.py --lr_schedule plateau [--epochs 50] [--batch_size 8] [--lr 1e-5]
"""

import os
import sys
import json
import argparse
import datetime

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt

# ── Local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from model   import Model2D
from dataset import build_tf_dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_ROOT = "/Volumes/Expansion1TB/MS/model_dataset"
RUNS_DIR     = os.path.join(os.path.dirname(__file__), "runs")


# ─────────────────────────────────────────────────────────────────────────────
# Custom Keras metrics
# ─────────────────────────────────────────────────────────────────────────────

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
        dice   = tf.reduce_mean((2.0 * inter + 1e-7) / (union + 1e-7))

        self.dice_sum.assign_add(dice)
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
        iou    = tf.reduce_mean((inter + 1e-7) / (union + 1e-7))

        self.iou_sum.assign_add(iou)
        self.batch_count.assign_add(1.0)

    def result(self):
        return self.iou_sum / (self.batch_count + 1e-7)

    def reset_state(self):
        self.iou_sum.assign(0.0)
        self.batch_count.assign(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Build and compile model
# ─────────────────────────────────────────────────────────────────────────────

def build_model(lr: float = 1e-5) -> tf.keras.Model:
    """
    Instantiate Model2D with multi-modal 3-channel input (FLAIR, T1w, T2w)
    and compile with Adam, BinaryCrossentropy, DiceScore, IoUScore.
    """
    m2d = Model2D(IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=3)
    m2d.initializeModel()

    # Recompile with richer metrics and a stable loss
    m2d.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            DiceScore(name="dice_score"),
            IoUScore(name="iou_score"),
        ],
    )
    return m2d.model


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history, run_dir: str):
    """
    Generate and save two plots:
      1. dice_curve.png  — Dice Coefficient vs Epochs (train + test)
      2. iou_curve.png   — IoU Score vs Epochs        (train + test)
    """
    epochs = range(1, len(history.history["loss"]) + 1)

    def _save(metric_key, val_metric_key, ylabel, title, filename):
        train_vals = history.history.get(metric_key, [])
        test_vals  = history.history.get(val_metric_key, [])

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, train_vals, "b-o",  markersize=4, linewidth=1.5, label="Train")
        ax.plot(epochs, test_vals,  "r--s", markersize=4, linewidth=1.5, label="Test (internal)")

        best_epoch = int(np.argmax(test_vals)) + 1 if test_vals else 0
        if test_vals:
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
        ax.set_ylim([0, 1])

        out_path = os.path.join(run_dir, filename)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  Saved plot: {out_path}")

    _save("dice_score", "val_dice_score",
          "Dice Coefficient",
          "2-D U-Net — Dice Coefficient vs Epochs (FLAIR + T1w + T2w)",
          "dice_curve.png")

    _save("iou_score", "val_iou_score",
          "IoU Score",
          "2-D U-Net — IoU Score vs Epochs (FLAIR + T1w + T2w)",
          "iou_curve.png")


def print_epoch_table(history):
    """
    Print a compact table of per-epoch train / test metrics to stdout.
    """
    h = history.history
    keys_train = ["loss", "dice_score", "iou_score", "accuracy"]
    keys_test  = [f"val_{k}" for k in keys_train]

    n_epochs = len(h["loss"])
    header = (f"{'Epoch':>6}  "
              f"{'Loss(tr)':>9} {'Dice(tr)':>9} {'IoU(tr)':>8} {'Acc(tr)':>8}  "
              f"{'Loss(ts)':>9} {'Dice(ts)':>9} {'IoU(ts)':>8} {'Acc(ts)':>8}")
    sep = "-" * len(header)

    print("\n" + sep)
    print("  PER-EPOCH RESULTS  (tr = train | ts = test/internal)")
    print(sep)
    print(header)
    print(sep)

    for ep in range(n_epochs):
        row = (f"{ep+1:>6}  "
               f"{h['loss'][ep]:>9.4f} "
               f"{h.get('dice_score', [0]*n_epochs)[ep]:>9.4f} "
               f"{h.get('iou_score',  [0]*n_epochs)[ep]:>8.4f} "
               f"{h.get('accuracy',   [0]*n_epochs)[ep]:>8.4f}  "
               f"{h.get('val_loss',   [0]*n_epochs)[ep]:>9.4f} "
               f"{h.get('val_dice_score', [0]*n_epochs)[ep]:>9.4f} "
               f"{h.get('val_iou_score',  [0]*n_epochs)[ep]:>8.4f} "
               f"{h.get('val_accuracy',   [0]*n_epochs)[ep]:>8.4f}")
        print(row)
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Print Layer Shapes for 1 Sample
# ─────────────────────────────────────────────────────────────────────────────

def print_layer_shapes(model, dataset):
    print("\n" + "="*60)
    print("Extracting intermediate layer shapes for 1 sample (Maxpool & Upsampled)...")
    print("="*60)
    # Filter max_pooling2d and conv2d_transpose layers
    target_layers = [l for l in model.layers if 'max_pooling2d' in l.name or 'conv2d_transpose' in l.name]
    
    if not target_layers:
        print("No maxpool or upsample layers found by name.")
        return
        
    int_model = tf.keras.Model(inputs=model.inputs, outputs=[l.output for l in target_layers])
    
    for x, y in dataset.take(1):
        sample = x[0:1] # take 1 sample
        preds = int_model.predict(sample, verbose=0)
        print(f"{'Input Layer':25s} shape: {sample.shape}")
        for l, p in zip(target_layers, preds):
            layer_type = "Maxpool" if 'max' in l.name else "Upsample"
            print(f"[{layer_type}] {l.name:15s} output shape: {p.shape}")
        break
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(epochs: int = 50,
          batch_size: int = 8,
          lr: float = 1e-5,
          lr_schedule: str = "fixed",
          skip_blank_ratio: float = 0.95):
    """
    Args:
        lr_schedule       : 'fixed'   – constant LR (default)
                            'plateau' – ReduceLROnPlateau on val_dice_score
        skip_blank_ratio  : Fraction of background-only training slices to drop.
                            0.0 = keep all slices; 0.95 = drop 95% (default).
                            Validation and test splits always keep all slices.
    """
    run_id  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nRun directory: {run_dir}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    # Training split: blank-slice filtering applied
    train_ds, n_train = build_tf_dataset(
        os.path.join(DATASET_ROOT, "train"),
        batch_size=batch_size,
        is_train=True,
        skip_blank_ratio=skip_blank_ratio,
    )
    # Validation during fit: ISBI2015 external set (monitors generalisation)
    # All slices kept (skip_blank_ratio=0.0)
    val_ds, n_val = build_tf_dataset(
        os.path.join(DATASET_ROOT, "external_val"),
        batch_size=batch_size,
        is_train=False,
        skip_blank_ratio=0.0,
    )
    # Held-out test: MSLesSeg 10% patients – NOT seen during training
    # All slices kept (skip_blank_ratio=0.0)
    test_ds, n_test = build_tf_dataset(
        os.path.join(DATASET_ROOT, "test"),
        batch_size=batch_size,
        is_train=False,
        skip_blank_ratio=0.0,
    )

    blank_label = f"{skip_blank_ratio:.0%}" if skip_blank_ratio > 0 else "none (keep all)"
    print(f"\nSlice counts  train={n_train}  val(ISBI)={n_val}  test(MSLesSeg)={n_test}")
    print(f"Blank-slice filtering : {blank_label} of background slices dropped from training")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(lr=lr)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(run_dir, "best_model.keras")

    callbacks = [
        # Save best weights monitored on ISBI2015 validation Dice
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_dice_score",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_score",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        # CSV log
        tf.keras.callbacks.CSVLogger(
            os.path.join(run_dir, "training_log.csv")
        ),
    ]

    # ── Optional: ReduceLROnPlateau ───────────────────────────────────────────
    if lr_schedule == "plateau":
        print(f"  LR schedule : ReduceLROnPlateau  (initial lr={lr}, factor=0.5, patience=5, min_lr=1e-7)")
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_dice_score",
                mode="max",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
            )
        )
    else:
        print(f"  LR schedule : fixed  (lr={lr})")

    # ── Print intermediate layer shapes for 1 sample ──────────────────────────
    print_layer_shapes(model, train_ds)

    # ── Fit (val = ISBI2015, used to monitor generalisation each epoch) ─────────
    print(f"\nStarting training for up to {epochs} epochs ...")
    print(f"  Validation : ISBI2015 external set (all slices, no blank filtering)")
    print(f"  Test set   : MSLesSeg 10% patients (evaluated once AFTER training)\n")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Per-epoch table ───────────────────────────────────────────────────────
    print_epoch_table(history)

    # ── Training curves ───────────────────────────────────────────────────────
    print("\nSaving training curve plots ...")
    plot_training_curves(history, run_dir)

    # ── Held-out test: MSLesSeg 10% patients (evaluated ONCE, post-training) ──
    print("\n" + "="*60)
    print("Evaluating on held-out MSLesSeg test set (10% patients) ...")
    print("="*60)
    test_results = model.evaluate(test_ds, verbose=1)
    test_metrics = dict(zip(model.metrics_names, test_results))

    # ── Best epoch stats from ISBI2015 validation ─────────────────────────────
    best_val_dice = max(history.history.get("val_dice_score", [0]))
    best_val_iou  = max(history.history.get("val_iou_score",  [0]))

    summary = {
        "run_id": run_id,
        "lr": lr,
        "lr_schedule": lr_schedule,
        "skip_blank_ratio": skip_blank_ratio,
        "epochs_trained": len(history.history["loss"]),
        "val_isbi2015": {
            "best_dice": round(float(best_val_dice), 4),
            "best_iou":  round(float(best_val_iou),  4),
        },
        "test_mslesseg": {
            k: round(float(v), 4) for k, v in test_metrics.items()
        },
        "checkpoint": ckpt_path,
        "training_log": os.path.join(run_dir, "training_log.csv"),
    }

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("TRAINING COMPLETE – Final Metrics")
    print("="*60)
    print("  Validation  (ISBI2015 – best epoch during training):")
    print(f"    Dice : {best_val_dice:.4f}")
    print(f"    IoU  : {best_val_iou:.4f}")
    print("  Test  (MSLesSeg 10% – evaluated once after training):")
    print(f"    Dice : {test_metrics.get('dice_score', 'N/A')}")
    print(f"    IoU  : {test_metrics.get('iou_score',  'N/A')}")
    print(f"\nSummary saved to: {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 2-D U-Net for MS lesion segmentation")
    parser.add_argument("--epochs",      type=int,   default=50,   help="Max training epochs")
    parser.add_argument("--batch_size",  type=int,   default=8,    help="Batch size")
    parser.add_argument("--lr",          type=float, default=1e-5, help="Learning rate (default: 1e-5)")
    parser.add_argument(
        "--lr_schedule",
        type=str,
        choices=["fixed", "plateau"],
        default="fixed",
        help=(
            "Learning rate schedule: "
            "'fixed' = constant LR throughout (default); "
            "'plateau' = ReduceLROnPlateau on val_dice_score "
            "(factor=0.5, patience=5, min_lr=1e-7)"
        ),
    )
    parser.add_argument(
        "--skip_blank_ratio",
        type=float,
        default=0.95,
        metavar="[0.0-1.0]",
        help=(
            "Fraction of lesion-free (background-only) training slices to discard. "
            "0.0 = keep all slices (no filtering); "
            "0.95 = drop 95%% of blank slices (default). "
            "Val and test splits are never filtered."
        ),
    )
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        skip_blank_ratio=args.skip_blank_ratio,
    )
