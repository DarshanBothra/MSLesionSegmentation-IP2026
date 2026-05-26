import os
import glob
import random
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
    print("Processing ISBI2015 Dataset (Preprocessed Data)...")
    train_dir = os.path.join(data_dir, 'training')
    if not os.path.exists(train_dir):
        print(f"Directory {train_dir} does not exist.")
        return False
    
    subdirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and d.startswith('training')]
    
    # Select 5 random samples
    random.seed(42) # Set seed for reproducibility, or remove to be completely random
    selected_subdirs = random.sample(subdirs, min(5, len(subdirs)))
    
    # 5 samples, 4 columns (T1(mprage), T2, FLAIR, MASK)
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    fig.suptitle('ISBI2015 Dataset (Preprocessed) - 5 Random Samples (T1, T2, FLAIR, Mask)', fontsize=20)
    
    for i, subdir in enumerate(selected_subdirs):
        # Paths for Preprocessed data
        mprage_file = glob.glob(os.path.join(subdir, 'preprocessed', '*_mprage_pp.nii'))
        t2_file = glob.glob(os.path.join(subdir, 'preprocessed', '*_t2_pp.nii'))
        flair_file = glob.glob(os.path.join(subdir, 'preprocessed', '*_flair_pp.nii'))
        mask_file = glob.glob(os.path.join(subdir, 'masks', '*_mask1.nii'))
        
        if not (mprage_file and t2_file and flair_file and mask_file):
            print(f"Skipping {subdir} - Missing one of the required files.")
            continue
            
        t1_slice, t1_shape = extract_middle_slice(mprage_file[0])
        t2_slice, t2_shape = extract_middle_slice(t2_file[0])
        flair_slice, flair_shape = extract_middle_slice(flair_file[0])
        mask_slice, mask_shape = extract_middle_slice(mask_file[0])
        
        patient_id = os.path.basename(subdir)
        
        # Plot T1 (MPRAGE)
        axes[i, 0].imshow(t1_slice, cmap='gray')
        axes[i, 0].set_title(f"Patient: {patient_id}\nT1w (MPRAGE) Preprocessed\nShape: {t1_shape}")
        axes[i, 0].axis('off')
        
        # Plot T2
        axes[i, 1].imshow(t2_slice, cmap='gray')
        axes[i, 1].set_title(f"T2w Preprocessed\nShape: {t2_shape}")
        axes[i, 1].axis('off')

        # Plot FLAIR
        axes[i, 2].imshow(flair_slice, cmap='gray')
        axes[i, 2].set_title(f"FLAIR Preprocessed\nShape: {flair_shape}")
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
    print("Processing MSLesSeg Dataset (Preprocessed)...")
    train_dir = os.path.join(mslesseg_dir, 'MSLesSeg Dataset', 'train')
    
    if not os.path.exists(train_dir):
        print(f"Directory {train_dir} does not exist.")
        return False
        
    subdirs = [os.path.join(train_dir, d) for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and d.startswith('P')]
    
    random.seed(42) # Set seed for reproducibility
    selected_subdirs = random.sample(subdirs, min(5, len(subdirs)))
    
    # 5 samples, 4 columns (T1, T2, FLAIR, MASK)
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    fig.suptitle('MSLesSeg Dataset (Preprocessed) - 5 Random Samples (T1, T2, FLAIR, Mask)', fontsize=20)
    
    for i, subdir in enumerate(selected_subdirs):
        p_id = os.path.basename(subdir)
        t_dirs = [os.path.join(subdir, d) for d in os.listdir(subdir) if os.path.isdir(os.path.join(subdir, d))]
        if not t_dirs:
            continue
        first_t_dir = sorted(t_dirs)[0]
        t_id = os.path.basename(first_t_dir)
        
        # Images from Preprocessed Dataset
        t1_file = glob.glob(os.path.join(first_t_dir, '*_T1.nii.gz'))
        t2_file = glob.glob(os.path.join(first_t_dir, '*_T2.nii.gz'))
        flair_file = glob.glob(os.path.join(first_t_dir, '*_FLAIR.nii.gz'))
        mask_file = glob.glob(os.path.join(first_t_dir, '*_MASK.nii.gz'))
        
        if not (t1_file and t2_file and flair_file and mask_file):
            print(f"Skipping {subdir} - Missing one of the required files.")
            continue
            
        t1_slice, t1_shape = extract_middle_slice(t1_file[0])
        t2_slice, t2_shape = extract_middle_slice(t2_file[0])
        flair_slice, flair_shape = extract_middle_slice(flair_file[0])
        mask_slice, mask_shape = extract_middle_slice(mask_file[0])
        
        # Plot T1
        axes[i, 0].imshow(t1_slice, cmap='gray')
        axes[i, 0].set_title(f"Patient: {p_id} {t_id}\nT1w Preprocessed\nShape: {t1_shape}")
        axes[i, 0].axis('off')
        
        # Plot T2
        axes[i, 1].imshow(t2_slice, cmap='gray')
        axes[i, 1].set_title(f"T2w Preprocessed\nShape: {t2_shape}")
        axes[i, 1].axis('off')

        # Plot FLAIR
        axes[i, 2].imshow(flair_slice, cmap='gray')
        axes[i, 2].set_title(f"FLAIR Preprocessed\nShape: {flair_shape}")
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
    
    # ISBI2015 (Using preprocessed)
    process_isbi2015(os.path.join(data_dir, 'ISBI2015'), os.path.join(out_dir, 'isbi2015_preprocessed.png'))
    
    # MSLesSeg (Using preprocessed dataset)
    process_mslesseg(os.path.join(data_dir, 'MSLesSeg'), os.path.join(out_dir, 'mslesseg_preprocessed.png'))
    
    print("Preprocessed Data EDA Complete. Images saved to", out_dir)
