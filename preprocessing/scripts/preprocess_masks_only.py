import os
import json
import ants
import numpy as np
from tqdm import tqdm

TEMPLATE_PATH = "/home/darshan/MS/preprocessing/TEMPLATE/MNI152_T1_1mm.nii.gz"
SAMPLES_JSON = "/home/darshan/MS/preprocessing/samples.json"
OUTPUT_BASE = "/home/darshan/MS/data/PREPROCESSED256"

def process_subject_mask(subject_name, subject_data):
    try:
        t1_path = subject_data["t1"]
        flair_path = subject_data["flair"]
        mask_path = subject_data["mask"]
        dataset_name = subject_data["dataset"].lower()
        
        mask_output = os.path.join(OUTPUT_BASE, subject_name, "mask.nii.gz")
        
        # Load MNI template
        mni = ants.image_read(TEMPLATE_PATH)
        target_orientation = mni.orientation
        
        # Load raw mask and reorient to RAS standard
        mask = ants.image_read(mask_path)
        mask = ants.reorient_image2(mask, target_orientation)
        
        if dataset_name == "miccai":
            # Miccai requires co-registration to T1 and then affine registration to MNI
            t1 = ants.image_read(t1_path)
            t1 = ants.reorient_image2(t1, target_orientation)
            
            flair = ants.image_read(flair_path)
            flair = ants.reorient_image2(flair, target_orientation)
            
            flair_to_t1 = ants.registration(fixed=t1, moving=flair, type_of_transform='Rigid')
            t1_to_mni = ants.registration(fixed=mni, moving=t1, type_of_transform='Affine')
            
            mask_mni = ants.apply_transforms(
                fixed=mni,
                moving=mask,
                transformlist=[t1_to_mni['fwdtransforms'][0], flair_to_t1['fwdtransforms'][0]],
                interpolator='genericLabel'
            )
        else:
            # ISBI and MSLesSeg are already registered, no registration needed
            mask_mni = mask
            
        # Resample to 256x256x182 (with nearestNeighbor interpolation to preserve binary labels)
        new_shape = (256, 256, mni.shape[2])
        resampled_mask = ants.resample_image(mask_mni, new_shape, use_voxels=True, interp_type=1)
        
        # Write preprocessed mask to disk
        os.makedirs(os.path.dirname(mask_output), exist_ok=True)
        ants.image_write(resampled_mask, mask_output)
        
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    with open(SAMPLES_JSON, 'r') as f:
        data = json.load(f)
        
    subjects = sorted(list(data.keys()))
    print(f"Total subjects to process: {len(subjects)}")
    print("Running sequentially to prevent CPU/RAM starvation...")
    
    success_count = 0
    fail_count = 0
    
    # Process subjects one-by-one to maintain lightweight footprint
    for subj in tqdm(subjects, desc="Processing Masks"):
        success, error = process_subject_mask(subj, data[subj])
        if success:
            success_count += 1
        else:
            fail_count += 1
            print(f"\nSubject {subj} failed: {error}")
            
    print(f"\nFinished preprocessing masks. Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    main()
