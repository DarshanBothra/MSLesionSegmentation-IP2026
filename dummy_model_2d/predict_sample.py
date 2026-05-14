import os
import argparse
import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_H, TARGET_W = 256, 256

def centre_pad_2d(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
    h, w = arr.shape
    pad_h = th - h
    pad_w = tw - w
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
        return None, None
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32), img

def predict_sample(t1_path, t2_path, flair_path, gt_path, model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    flair, flair_nii = load_nii(flair_path)
    t1, _ = load_nii(t1_path)
    t2, _ = load_nii(t2_path)
    gt_mask, _ = load_nii(gt_path)

    if flair is None:
        raise FileNotFoundError(f"FLAIR image not found at {flair_path}")

    H, W, D = flair.shape
    
    if t1 is None: t1 = np.zeros_like(flair)
    if t2 is None: t2 = np.zeros_like(flair)

    imgs_list = []
    for z in range(D):
        f_s  = centre_pad_2d(flair[:, :, z], TARGET_H, TARGET_W)
        t1_s = centre_pad_2d(t1[:, :, z], TARGET_H, TARGET_W)
        t2_s = centre_pad_2d(t2[:, :, z], TARGET_H, TARGET_W)
        imgs_list.append(np.stack([f_s, t1_s, t2_s], axis=-1))

    X = np.array(imgs_list, dtype=np.float32)

    print(f"Loading model from {model_path} ...")
    # Custom objects might be needed for dice_loss
    import segmentation_models as sm
    model = tf.keras.models.load_model(model_path, custom_objects={'dice_loss': sm.losses.dice_loss}, compile=False)

    print("Running inference...")
    preds = model.predict(X, batch_size=8, verbose=1)
    preds_bin = (preds > 0.5).astype(np.float32)[..., 0]

    pred_vol = np.zeros((H, W, D), dtype=np.float32)
    for z in range(D):
        pred_vol[:, :, z] = unpad_2d(preds_bin[z], H, W)
        
    out_mask_path = os.path.join(output_dir, "predicted_mask.nii.gz")
    out_nii = nib.Nifti1Image(pred_vol, flair_nii.affine, flair_nii.header)
    nib.save(out_nii, out_mask_path)
    print(f"Saved predicted mask volume to {out_mask_path}")

    # Plot middle slice
    mid_z = D // 2
    print(f"Plotting middle slice (z={mid_z}) ...")
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # 17. Mask should be: lesion pixels white, background black. cmap='gray' achieves this.
    axes[0].imshow(np.rot90(t1[:, :, mid_z]), cmap='gray')
    axes[0].set_title('T1w')
    axes[0].axis('off')
    
    axes[1].imshow(np.rot90(t2[:, :, mid_z]), cmap='gray')
    axes[1].set_title('T2w')
    axes[1].axis('off')
    
    axes[2].imshow(np.rot90(flair[:, :, mid_z]), cmap='gray')
    axes[2].set_title('FLAIR')
    axes[2].axis('off')
    
    if gt_mask is not None:
        axes[3].imshow(np.rot90(gt_mask[:, :, mid_z] > 0), cmap='gray')
        axes[3].set_title('Ground Truth Mask')
    else:
        axes[3].imshow(np.zeros((H, W)), cmap='gray')
        axes[3].set_title('No GT Mask')
    axes[3].axis('off')
    
    axes[4].imshow(np.rot90(pred_vol[:, :, mid_z]), cmap='gray')
    axes[4].set_title('Predicted Mask')
    axes[4].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "middle_slice_prediction.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved middle slice plot to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict mask from T1, T2, FLAIR and print middle slice")
    parser.add_argument("--t1", default=None, help="Path to T1w .nii.gz")
    parser.add_argument("--t2", default=None, help="Path to T2w .nii.gz")
    parser.add_argument("--flair", required=True, help="Path to FLAIR .nii.gz")
    parser.add_argument("--gt", default=None, help="Path to Ground Truth mask .nii.gz (optional)")
    parser.add_argument("--model", required=True, help="Path to trained model (.h5)")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    
    args = parser.parse_args()
    predict_sample(args.t1, args.t2, args.flair, args.gt, args.model, args.output_dir)
