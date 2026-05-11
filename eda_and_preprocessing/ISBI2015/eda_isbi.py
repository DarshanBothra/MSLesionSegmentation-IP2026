import os
import nibabel as nib
import numpy as np

def analyze_isbi_directory(base_dir, split_type):
    print(f"==================================================")
    print(f"Analyzing ISBI2015 {split_type} ({base_dir})")
    print(f"==================================================")
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.\n")
        return

    subjects = [d for d in os.listdir(base_dir) if d.startswith('training') or d.startswith('test')]
    subjects = [d for d in subjects if os.path.isdir(os.path.join(base_dir, d))]
    
    print(f"Total subjects found: {len(subjects)}")
    
    for folder_type in ['orig', 'preprocessed']:
        total_t1 = 0
        total_t2 = 0
        total_pd = 0
        total_flair = 0
        total_masks = 0
        
        control_cases = 0
        lesion_cases = 0
        
        print(f"\n--- {folder_type.upper()} IMAGES ---")
        
        for subject in sorted(subjects):
            subj_path = os.path.join(base_dir, subject)
            img_path = os.path.join(subj_path, folder_type)
            mask_path = os.path.join(subj_path, 'masks')
            
            if not os.path.exists(img_path):
                continue
                
            img_files = sorted(os.listdir(img_path))
            shapes = []
            
            # Find unique timepoints
            timepoints = set()
            for f in img_files:
                if f.endswith('.nii') or f.endswith('.nii.gz'):
                    parts = f.split('_')
                    if len(parts) >= 2:
                        timepoints.add(parts[1])
            
            for tp in sorted(list(timepoints)):
                tp_img_files = [f for f in img_files if f"_{tp}_" in f]
                tp_shapes = []
                
                for f in tp_img_files:
                    f_path = os.path.join(img_path, f)
                    try:
                        img = nib.load(f_path)
                        tp_shapes.append((f, img.shape))
                        if 'mprage' in f:
                            total_t1 += 1
                        elif 't2' in f:
                            total_t2 += 1
                        elif 'pd' in f:
                            total_pd += 1
                        elif 'flair' in f:
                            total_flair += 1
                    except Exception as e:
                        print(f"Error loading {f_path}: {e}")
                
                # Check masks for this timepoint
                has_mask = False
                mask_has_lesion = False
                if os.path.exists(mask_path):
                    mask_files = [f for f in os.listdir(mask_path) if f"_{tp}_mask" in f and (f.endswith('.nii') or f.endswith('.nii.gz'))]
                    for mf in mask_files:
                        total_masks += 1
                        has_mask = True
                        try:
                            m_img = nib.load(os.path.join(mask_path, mf))
                            if np.any(m_img.get_fdata() > 0):
                                mask_has_lesion = True
                        except:
                            pass
                
                if tp_shapes:
                    print(f"[{subject} - {tp}] Images: {', '.join([f'{name}: {s}' for name, s in tp_shapes])}")
                if has_mask:
                    if mask_has_lesion:
                        lesion_cases += 1
                    else:
                        control_cases += 1

        print(f"\nSummary for {split_type} ({folder_type}):")
        print(f"Total MPRAGE (T1): {total_t1}")
        print(f"Total T2: {total_t2}")
        print(f"Total PD: {total_pd}")
        print(f"Total FLAIR: {total_flair}")
        print(f"Total Masks (Rater 1 + Rater 2): {total_masks}")
        if total_masks > 0:
            print(f"Control cases (All masks empty): {control_cases}")
            print(f"Lesion cases (Some mask has lesions): {lesion_cases}")

if __name__ == '__main__':
    data_dir = '/Volumes/Expansion1TB/MS/data/ISBI2015'
    
    # Analyze training
    analyze_isbi_directory(os.path.join(data_dir, 'training'), 'Training')
    
    # Analyze test
    analyze_isbi_directory(os.path.join(data_dir, 'testdata_website'), 'Test')
