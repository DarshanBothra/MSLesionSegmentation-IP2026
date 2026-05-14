import os
import argparse
import numpy as np
import nibabel as nib
import tensorflow as tf

def centre_pad_2d(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Zero-pad a 2-D array to (th, tw), centred."""
    h, w = arr.shape
    pad_h = th - h
    pad_w = tw - w
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left
    return np.pad(arr, ((top, bottom), (left, right)), mode="constant")

def unpad_2d(arr: np.ndarray, original_h: int, original_w: int) -> np.ndarray:
    """Unpad a 2-D array to original size, assuming it was centre-padded."""
    h, w = arr.shape
    pad_h = h - original_h
    pad_w = w - original_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    if bottom == 0 and right == 0:
        return arr[top:, left:]
    elif bottom == 0:
        return arr[top:, left:-right]
    elif right == 0:
        return arr[top:-bottom, left:]
    return arr[top:-bottom, left:-right]

def predict(input_path, model_path, output_path):
    print(f"Loading input image from {input_path}...")
    img_nii = nib.load(input_path)
    img_data = img_nii.get_fdata(dtype=np.float32)
    
    # Image should be (H, W, D) -> e.g. (182, 218, 182)
    H, W, D = img_data.shape
    TARGET_H, TARGET_W = 256, 256
    
    print(f"Original shape: {H}x{W}x{D}")
    print("Preparing slices...")
    
    imgs_list = []
    for z in range(D):
        slice_data = img_data[:, :, z]
        
        # Pad FLAIR slice
        f_s = centre_pad_2d(slice_data, TARGET_H, TARGET_W)
        
        # Fill T1 and T2 with zeros (as model expects 3 channels)
        t1_s = np.zeros_like(f_s)
        t2_s = np.zeros_like(f_s)
        
        # Stack into (256, 256, 3)
        img_slice = np.stack([f_s, t1_s, t2_s], axis=-1)
        imgs_list.append(img_slice)
        
    X = np.array(imgs_list, dtype=np.float32)
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    print("Running inference...")
    preds = model.predict(X, batch_size=8, verbose=1)
    
    # Thresholding at 0.5
    preds_bin = (preds > 0.5).astype(np.float32) # Shape: (D, 256, 256, 1)
    preds_bin = preds_bin[..., 0] # Shape: (D, 256, 256)
    
    print("Reconstructing volume...")
    out_mask = np.zeros((H, W, D), dtype=np.float32)
    for z in range(D):
        padded_mask = preds_bin[z]
        # Unpad back to original (H, W)
        unpadded = unpad_2d(padded_mask, H, W)
        out_mask[:, :, z] = unpadded
        
    print(f"Saving predicted mask to {output_path}...")
    out_nii = nib.Nifti1Image(out_mask, img_nii.affine, img_nii.header)
    nib.save(out_nii, output_path)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict mask for a single MRI volume")
    parser.add_argument("--input", required=True, help="Path to input .nii.gz image")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint (.keras)")
    parser.add_argument("--output", required=True, help="Path to save the predicted mask (.nii.gz)")
    args = parser.parse_args()
    
    predict(args.input, args.model, args.output)
