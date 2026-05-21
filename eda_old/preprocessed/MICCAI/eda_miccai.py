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

    patients = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Total patients found: {len(patients)}")
    
    total_flair_1 = 0
    total_flair_2 = 0
    total_masks_expert_1 = 0
    total_masks_expert_2 = 0
    total_masks_expert_3 = 0
    total_masks_expert_4 = 0
    total_ground_truth_masks = 0

    for patient in sorted(patients):
        patient_path = os.path.join(base_dir, patient)
            
        files = os.listdir(patient_path)
        mask_has_lesions = False
        shapes = []
        for f in files:
            if not (f.endswith('.nii') or f.endswith('.nii.gz')):
                continue
                
            f_path = os.path.join(patient_path, f)
            try:
                img = nib.load(f_path)
                shape = img.shape
                shapes.append((f, shape))
            except Exception as e:
                print(f"Error loading {f_path}: {e}")
                continue
                
            if 'EXPERT1' in f.upper():
                total_masks_expert_1 += 1
                data = img.get_fdata()
                if np.any(data > 0):
                    mask_has_lesions = True
            elif 'EXPERT2' in f.upper():
                total_masks_expert_2 += 1
                data = img.get_fdata()
                if np.any(data > 0):
                    mask_has_lesions = True
            elif 'EXPERT3' in f.upper():
                total_masks_expert_3 += 1
                data = img.get_fdata()
                if np.any(data > 0):
                    mask_has_lesions = True
            elif 'EXPERT4' in f.upper():
                total_masks_expert_4 += 1
                data = img.get_fdata()
                if np.any(data > 0):
                    mask_has_lesions = True
            elif 'TIME01' in f.upper():
                total_flair_1 += 1
            elif 'TIME02' in f.upper():
                total_flair_2 += 1
            else:
                total_ground_truth_masks += 1

    lesion_cases = max(total_masks_expert_1, total_masks_expert_2, total_masks_expert_3, total_masks_expert_4)
    non_lesion_cases = total_ground_truth_masks - lesion_cases

    print(f"Lesion Cases: {lesion_cases}")
    print(f"Non Lesion cases: {non_lesion_cases}")
    print("")

    print("\n--------------------------------------------------")
    print(f"Summary for {base_dir}:")
    print(f"Total FLAIR TIME 1 images: {total_flair_1}")
    print(f"Total FLAIR TIME 3 images: {total_flair_2}")
    print(f"Total Masks: {total_ground_truth_masks}")
    print(f"Control cases (Mask is completely empty/zero): {non_lesion_cases}")
    print(f"Lesion cases (Mask has >0 voxels): {lesion_cases}")
    print("\n")

if __name__ == '__main__':
    data_dir = '/home/darshan/MS/data/LongitudinalMultipleSclerosisLesionSegmentationChallengeMiccai21'
    
    
    # Analyze the MICCAI Dataset
    train_dir = os.path.join(data_dir, 'training')
    analyze_directory(train_dir)
