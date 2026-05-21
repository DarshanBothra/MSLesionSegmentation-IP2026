import os
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import random
import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import segmentation_models as sm

TARGET_H, TARGET_W = 256, 256

def centre_pad_2d(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
    h, w = arr.shape
    pad_h = max(0, th - h)
    pad_w = max(0, tw - w)
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left
    return np.pad(arr, ((top, bottom), (left, right)), mode="constant")

def unpad_2d(arr: np.ndarray, original_h: int, original_w: int) -> np.ndarray:
    h, w = arr.shape
    pad_h = h - original_h
    pad_w = w - original_w
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left

    h_end = h - bottom if bottom > 0 else h
    w_end = w - right  if right  > 0 else w
    return arr[top:h_end, left:w_end]

def load_nii(path):
    if not os.path.exists(path):
        return None
    data = nib.load(path).get_fdata(dtype=np.float32)
    # Min-Max Normalization
    dmin, dmax = data.min(), data.max()
    if dmax > dmin:
        data = (data - dmin) / (dmax - dmin)
    else:
        data = np.zeros_like(data)
    return data

def main():
    test_dir = '/home/darshan/MS/dummy_model_2d/dataset/split/test'
    model_path = '/home/darshan/MS/dummy_model_2d/runs/model4/best_model.h5'
    artifacts_dir = '/home/darshan/.gemini/antigravity/brain/2324e7f3-93f8-49b4-84ad-4558dc46bac9/artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Custom objects might be needed for bce_dice_loss
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={'dice_loss': sm.losses.dice_loss}, 
        compile=False
    )
    
    patients = [d for d in os.listdir(test_dir) if d.endswith("FLAIR.nii.gz")]
    # random.seed(42)  # For reproducibility
    # selected_patients = random.sample(patients, min(5, len(patients)))
    selected_patients = patients
    
    details = []
    
    for i, patient in enumerate(selected_patients):
        patient_path = os.path.join(test_dir, patient)
        
        flair = load_nii(patient_path)
        mask_path = patient_path.replace("FLAIR.nii.gz", "MASK.nii.gz")
        mask = nib.load(mask_path).get_fdata(dtype=np.float32) if os.path.exists(mask_path) else None
        
        if flair is None:
            continue
            
        H, W, D = flair.shape
        if mask is not None: mask = (mask > 0.5).astype(np.float32)
        else: mask = np.zeros_like(flair)
        
        # Find middle slice with lesion
        mid_z = D // 2
        best_z = mid_z
        if mask is not None and mask[:, :, mid_z].max() == 0:
            # Search outward
            found = False
            for offset in range(1, D // 2):
                if mid_z + offset < D and mask[:, :, mid_z + offset].max() > 0:
                    best_z = mid_z + offset
                    found = True
                    break
                if mid_z - offset >= 0 and mask[:, :, mid_z - offset].max() > 0:
                    best_z = mid_z - offset
                    found = True
                    break
            if not found:
                best_z = mid_z # Fallback
                
        # Prepare the slice for prediction
        f_s = centre_pad_2d(flair[:, :, best_z], TARGET_H, TARGET_W)
        
        x = np.stack([f_s], axis=-1)
        x_batch = np.expand_dims(x, axis=0) # (1, 256, 256, 1)
        
        # Predict
        pred = model.predict(x_batch, verbose=0)
        pred_bin = (pred[0, :, :, 0] > 0.5).astype(np.float32)
        
        # Unpad
        pred_unpad = unpad_2d(pred_bin, H, W)
        
        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(20, 4))
        
        axes[0].imshow(np.rot90(flair[:, :, best_z]), cmap='gray')
        axes[0].set_title('FLAIR')
        axes[0].axis('off')
        
        axes[1].imshow(np.rot90(mask[:, :, best_z]), cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        axes[2].imshow(np.rot90(pred_unpad), cmap='gray')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
        
        fig.suptitle(f"Patient: {patient} | Slice: {best_z} | Original Dim: {H}x{W}x{D} | Padded Dim: 256x256x{D}", fontsize=14)
        
        plt.tight_layout()
        plot_path = os.path.join("/home/darshan/MS/dummy_model_2d/test_predictions/model4/", f"{patient}_prediction.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        
        details.append({
            "patient": patient,
            "slice": best_z,
            "dim": f"{H}x{W}x{D}",
            "plot_path": plot_path
        })
        
    with open(os.path.join(artifacts_dir, 'prediction_details.json'), 'w') as f:
        json.dump(details, f, indent=2)

if __name__ == "__main__":
    main()
