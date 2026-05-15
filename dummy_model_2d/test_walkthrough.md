# Evaluating the Best Model on Test Samples

This walkthrough demonstrates how to use the specific model (`20260515_010229/best_model.h5`) to generate visual predictions on 5 random subjects from the completely held-out test dataset.

## Prerequisite

Make sure you are within your deep learning environment and that you are in the `dummy_model_2d` directory.

## 1. Running the Generation Script

I have created a script called `generate_test_results.py` that handles everything:
- It randomly selects 5 subjects from `/home/darshan/MS/model_dataset/test`.
- It dynamically searches for the middle slice of each subject's volume. If the exact middle slice has no lesion in the ground truth mask, it searches outwards to find the nearest slice that does.
- It applies the correct padding, uses `best_model.h5` to predict the mask, and removes the padding.
- It outputs a detailed 5-panel PNG image for each subject containing: **T1w**, **T2w**, **FLAIR**, **Ground Truth Mask**, and **Predicted Mask**.

Run the following command in your terminal:
```bash
python3 generate_test_results.py
```

## 2. Viewing the Results

Once the script completes, a new directory called `test_predictions` will be created inside `dummy_model_2d`:

```bash
ls -l test_predictions/
```

Inside this directory, you will find:
1. **Five `.png` files** (e.g., `P12_T1_prediction.png`):
   - You can open these images to inspect the side-by-side comparison. The mask displays lesion pixels in white and background in black.
   - The title of each image includes the Patient ID, the specific Slice Number, the Original Dimension (e.g., `182x218x182`), and the Padded Dimension (`256x256x182`).

2. **`prediction_details.json`**:
   - This JSON file logs the exact metadata (subject IDs, dimensions, slice numbers, and file paths) for programmatic reference.

You can inspect the JSON details directly:
```bash
cat test_predictions/prediction_details.json
```
