#!/bin/bash
# run.sh - Prepares dataset and trains the model

set -e

source /home/darshan/MS/venv/bin/activate
echo "Step 1: Preparing dataset..."
python3 prepare_dataset.py | tee /home/darshan/MS/logs/prepare_dataset.log

echo "Step 2: Training the model..."
python3 train.py --epochs 100 --batch_size 4 --lr 1e-5 --skip_blank_ratio 0.8 | tee /home/darshan/MS/logs/train.log

echo "Training completed."
