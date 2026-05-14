#!/bin/bash
# eval.sh — Run predict.py on a sample study from the training dataset
#            using the latest trained model checkpoint.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Find the latest run with a best_model.h5 ─────────────────────────────────
MODEL_PATH=""
for d in $(ls -td runs/*/); do
    if [ -f "${d}best_model.h5" ]; then
        MODEL_PATH="${d}best_model.h5"
        break
    fi
done

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: No best_model.h5 found in any runs/ subdirectory."
    exit 1
fi

echo "Using model: $MODEL_PATH"

# ── Dataset directory ─────────────────────────────────────────────────────────
DATASET_DIR="/home/darshan/MS/model_dataset/train"

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory $DATASET_DIR not found."
    echo "Make sure you have run prepare_dataset.py first."
    exit 1
fi

# ── Pick a study with a mask (P1_T1 is a good default) ───────────────────────
STUDY="P1_T1"
STUDY_DIR="${DATASET_DIR}/${STUDY}"

if [ ! -d "$STUDY_DIR" ]; then
    # Fallback: pick the first available study
    STUDY=$(ls "$DATASET_DIR" | head -n 1)
    STUDY_DIR="${DATASET_DIR}/${STUDY}"
fi

echo "Study directory: $STUDY_DIR"

# ── Output goes next to the model checkpoint ──────────────────────────────────
RUN_DIR=$(dirname "$MODEL_PATH")
OUTPUT_PNG="${RUN_DIR}/prediction_${STUDY}.png"

echo "Running prediction + visualisation ..."
python3 predict.py \
    --study_dir "$STUDY_DIR" \
    --model     "$MODEL_PATH" \
    --output    "$OUTPUT_PNG"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Evaluation complete."
echo "  Figure : $OUTPUT_PNG"
echo "  Mask   : ${OUTPUT_PNG%.png}_pred_mask.nii.gz"
echo "═══════════════════════════════════════════════════════════"
