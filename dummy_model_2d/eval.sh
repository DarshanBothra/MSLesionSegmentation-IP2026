#!/bin/bash
# eval.sh - Uses the trained model to predict on an image from the training dataset

set -e

# Find the latest model run
LATEST_RUN=$(ls -td runs/*/ | head -1)
MODEL_PATH="${LATEST_RUN}best_model.keras"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model checkpoint not found in any runs/ subdirectory."
    exit 1
fi

echo "Using model: $MODEL_PATH"

# Find an image from the training dataset
DATASET_DIR="/Volumes/Expansion1TB/MS/model_dataset/train"
if [ ! -d "$DATASET_DIR" ]; then
    echo "Warning: Dataset directory $DATASET_DIR not found."
    echo "Please make sure you have run the dataset preparation step."
    exit 1
fi

INPUT_IMAGE=$(find "$DATASET_DIR" -name "flair.nii.gz" | head -n 1)

if [ -z "$INPUT_IMAGE" ]; then
    echo "Error: Could not find any flair.nii.gz image in $DATASET_DIR."
    exit 1
fi

OUTPUT_MASK="predicted_mask.nii.gz"

echo "Running prediction on $INPUT_IMAGE..."
python predict.py --input "$INPUT_IMAGE" --model "$MODEL_PATH" --output "$OUTPUT_MASK"

echo "Prediction complete. Output saved to $OUTPUT_MASK."
