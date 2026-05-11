import os
import nibabel as nib
import numpy as np

def analyze_directory(base_dir):
    print(f"==================================================")
    print(f"Analyzing {base_dir}")
    print(f"==================================================")
    
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist.\n")
        return

    patients = [d for d in os.listdir(base_dir) if d.startswith('P') and os.path.isdir(os.path.join(base_dir, d))]
    print(f"Total patients found: {len(patients)}")
    
    total_t1 = 0
    total_t2 = 0
    total_flair = 0
    total_masks = 0
    
    control_cases = 0
    lesion_cases = 0
    
    missing_masks = 0
    
    for patient in sorted(patients):
        patient_path = os.path.join(base_dir, patient)
        timepoints = [d for d in os.listdir(patient_path) if d.startswith('T') and os.path.isdir(os.path.join(patient_path, d))]
        
        # If no timepoints (like in the test set), treat the patient folder as the timepoint
        if not timepoints:
            timepoints = ['']
            patient_is_tp = True
        else:
            patient_is_tp = False
            
        for tp in sorted(timepoints):
            tp_path = os.path.join(patient_path, tp) if not patient_is_tp else patient_path
            files = os.listdir(tp_path)
            
            has_mask = False
            mask_has_lesions = False
            
            shapes = []
            
            for f in files:
                if not (f.endswith('.nii') or f.endswith('.nii.gz')):
                    continue
                    
                f_path = os.path.join(tp_path, f)
                try:
                    img = nib.load(f_path)
                    shape = img.shape
                    shapes.append((f, shape))
                except Exception as e:
                    print(f"Error loading {f_path}: {e}")
                    continue
                    
                if 'MASK' in f.upper():
                    total_masks += 1
                    has_mask = True
                    data = img.get_fdata()
                    if np.any(data > 0):
                        mask_has_lesions = True
                elif 'FLAIR' in f.upper():
                    total_flair += 1
                elif 'T2' in f.upper():
                    total_t2 += 1
                elif 'T1' in f.upper():
                    total_t1 += 1

            if shapes:
                tp_name = tp if tp else "Study"
                print(f"[{patient} - {tp_name}] Images: {', '.join([f'{name}: {s}' for name, s in shapes])}")

            if not has_mask and shapes:
                missing_masks += 1
                
            if has_mask:
                if mask_has_lesions:
                    lesion_cases += 1
                else:
                    control_cases += 1

    print("\n--------------------------------------------------")
    print(f"Summary for {base_dir}:")
    print(f"Total T1 images: {total_t1}")
    print(f"Total T2 images: {total_t2}")
    print(f"Total FLAIR images: {total_flair}")
    print(f"Total Masks: {total_masks}")
    print(f"Timepoints with missing masks: {missing_masks}")
    print(f"Control cases (Mask is completely empty/zero): {control_cases}")
    print(f"Lesion cases (Mask has >0 voxels): {lesion_cases}")
    print("\n")

if __name__ == '__main__':
    data_dir = '/Volumes/Expansion1TB/MS/data/MSLesSeg'
    
    # Analyze the MSLesSeg_RAW dataset
    raw_dir = os.path.join(data_dir, 'MSLesSeg_RAW')
    analyze_directory(raw_dir)
    
    # Analyze the MSLesSeg Dataset train subset
    train_dir = os.path.join(data_dir, 'MSLesSeg Dataset', 'train')
    analyze_directory(train_dir)
    
    # Analyze the MSLesSeg Dataset test subset
    test_dir = os.path.join(data_dir, 'MSLesSeg Dataset', 'test')
    analyze_directory(test_dir)
