import os
import glob
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

def extract_middle_slice(nii_path):
    img = nib.load(nii_path)
    data = img.get_fdata()
    shape = data.shape
    # Handle potentially 4D data (take first volume)
    if len(shape) > 3:
        data = data[:, :, :, 0]
    
    # Extract middle slice along z-axis (axial view typically)
    middle_idx = data.shape[2] // 2
    slice_2d = data[:, :, middle_idx]
    
    # Rotate 90 degrees if needed for better display
    slice_2d = np.rot90(slice_2d)
    return slice_2d, shape

def process_isbi2015(data_dir, output_file):
    print("Processing ISBI2015 Dataset (Original Data)...")
    train_dir = os.path.join(data_dir, 'training')
    if not os.path.exists(train_dir):
        print(f"Directory {train_dir} does not exist.")
        return False
    
    subdirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and d.startswith('training')]
    subdirs = sorted(subdirs)[:5]
    
    # 5 samples, 4 columns (T1(mprage), T2, FLAIR, MASK)
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    fig.suptitle('ISBI2015 Dataset (Raw/Orig) - 5 Training Samples (T1, T2, FLAIR, Mask Separately)', fontsize=20)
    
    for i, subdir in enumerate(subdirs):
        # Paths for Original data
        mprage_file = glob.glob(os.path.join(subdir, 'orig', '*_mprage.nii.gz')) # Used as T1
        t2_file = glob.glob(os.path.join(subdir, 'orig', '*_t2.nii.gz'))
        flair_file = glob.glob(os.path.join(subdir, 'orig', '*_flair.nii.gz'))
        mask_file = glob.glob(os.path.join(subdir, 'masks', '*_mask1.nii'))
        
        if not (mprage_file and t2_file and flair_file and mask_file):
            print(f"Skipping {subdir} - Missing one of the required files.")
            continue
            
        t1_slice, t1_shape = extract_middle_slice(mprage_file[0])
        t2_slice, t2_shape = extract_middle_slice(t2_file[0])
        flair_slice, flair_shape = extract_middle_slice(flair_file[0])
        mask_slice, mask_shape = extract_middle_slice(mask_file[0])
        
        # Plot T1 (MPRAGE)
        axes[i, 0].imshow(t1_slice, cmap='gray')
        axes[i, 0].set_title(f"T1w (MPRAGE) Orig\nShape: {t1_shape}")
        axes[i, 0].axis('off')
        
        # Plot T2
        axes[i, 1].imshow(t2_slice, cmap='gray')
        axes[i, 1].set_title(f"T2w Orig\nShape: {t2_shape}")
        axes[i, 1].axis('off')

        # Plot FLAIR
        axes[i, 2].imshow(flair_slice, cmap='gray')
        axes[i, 2].set_title(f"FLAIR Orig\nShape: {flair_shape}")
        axes[i, 2].axis('off')
        
        # Plot Mask
        axes[i, 3].imshow(mask_slice, cmap='gray')
        axes[i, 3].set_title(f"Mask\nShape: {mask_shape}")
        axes[i, 3].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file)
    plt.close()
    print(f"Saved {output_file}")
    return True

def process_mslesseg(mslesseg_dir, output_file):
    print("Processing MSLesSeg Dataset (RAW)...")
    raw_dir = os.path.join(mslesseg_dir, 'MSLesSeg_RAW')
    train_annot_dir = os.path.join(mslesseg_dir, 'MSLesSeg Dataset', 'train') # For masks
    
    if not os.path.exists(raw_dir):
        print(f"Directory {raw_dir} does not exist.")
        return False
        
    subdirs = [os.path.join(raw_dir, d) for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d)) and d.startswith('P')]
    subdirs = sorted(subdirs)[:5]
    
    # 5 samples, 4 columns (T1, T2, FLAIR, MASK)
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    fig.suptitle('MSLesSeg Dataset (RAW) - 5 Training Samples (T1, T2, FLAIR, Mask Separately)', fontsize=20)
    
    for i, subdir in enumerate(subdirs):
        p_id = os.path.basename(subdir)
        t_dirs = [os.path.join(subdir, d) for d in os.listdir(subdir) if os.path.isdir(os.path.join(subdir, d))]
        if not t_dirs:
            continue
        first_t_dir = sorted(t_dirs)[0]
        t_id = os.path.basename(first_t_dir)
        
        # Images from RAW
        t1_file = glob.glob(os.path.join(first_t_dir, '*_T1.nii.gz'))
        t2_file = glob.glob(os.path.join(first_t_dir, '*_T2.nii.gz'))
        flair_file = glob.glob(os.path.join(first_t_dir, '*_FLAIR.nii.gz'))
        
        # Mask from annotated Dataset
        annot_t_dir = os.path.join(train_annot_dir, p_id, t_id)
        mask_file = glob.glob(os.path.join(annot_t_dir, '*_MASK.nii.gz'))
        
        if not (t1_file and t2_file and flair_file and mask_file):
            print(f"Skipping {subdir} - Missing one of the required files.")
            continue
            
        t1_slice, t1_shape = extract_middle_slice(t1_file[0])
        t2_slice, t2_shape = extract_middle_slice(t2_file[0])
        flair_slice, flair_shape = extract_middle_slice(flair_file[0])
        mask_slice, mask_shape = extract_middle_slice(mask_file[0])
        
        # Plot T1
        axes[i, 0].imshow(t1_slice, cmap='gray')
        axes[i, 0].set_title(f"T1w RAW\nShape: {t1_shape}")
        axes[i, 0].axis('off')
        
        # Plot T2
        axes[i, 1].imshow(t2_slice, cmap='gray')
        axes[i, 1].set_title(f"T2w RAW\nShape: {t2_shape}")
        axes[i, 1].axis('off')

        # Plot FLAIR
        axes[i, 2].imshow(flair_slice, cmap='gray')
        axes[i, 2].set_title(f"FLAIR RAW\nShape: {flair_shape}")
        axes[i, 2].axis('off')
        
        # Plot Mask
        axes[i, 3].imshow(mask_slice, cmap='gray')
        axes[i, 3].set_title(f"Mask\nShape: {mask_shape}")
        axes[i, 3].axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file)
    plt.close()
    print(f"Saved {output_file}")
    return True

if __name__ == "__main__":
    data_dir = "/Volumes/Expansion1TB/MS/data"
    
    out_dir = "/Volumes/Expansion1TB/MS/eda_results"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. ISBI2015 (Using orig instead of preprocessed)
    process_isbi2015(os.path.join(data_dir, 'ISBI2015'), os.path.join(out_dir, 'isbi2015_multimodal.png'))
    
    # 2. MSLesSeg (Using RAW)
    process_mslesseg(os.path.join(data_dir, 'MSLesSeg'), os.path.join(out_dir, 'mslesseg_multimodal.png'))
    
    print("EDA Complete. Images saved to", out_dir)
