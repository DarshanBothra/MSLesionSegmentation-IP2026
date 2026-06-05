import os
import json
import ants
from tqdm import tqdm
import nibabel as nib
import numpy as np

def register(subject_name, subject_data, template_path, output_base):

    t1_path = subject_data["t1"]
    t2_path = subject_data["t2"]
    flair_path = subject_data["flair"]
    mask_path = subject_data["mask"]
    dataset_name = subject_data["dataset"]

    os.makedirs(os.path.join(output_base, subject_name), exist_ok=True)
    t1_output = os.path.join(output_base, subject_name, "t1.nii.gz")
    t2_output = os.path.join(output_base, subject_name, "t2.nii.gz")
    flair_output = os.path.join(output_base, subject_name, "flair.nii.gz")
    mask_output = os.path.join(output_base, subject_name, "mask.nii.gz")
    """Performs rigid co-registration and template affine registration using ANTsPy."""
    # 1. Load all images
    mni = ants.image_read(template_path)
    target_orientation = mni.orientation
    
    t1 = ants.image_read(t1_path)
    t2 = ants.image_read(t2_path)       
    flair = ants.image_read(flair_path)
    mask = ants.image_read(mask_path)

    # 2. Reorient all images to match the template standard (RAS)
    print(f"Reorienting to target orientation: {target_orientation}")
    t1 = ants.reorient_image2(t1, target_orientation)
    t2 = ants.reorient_image2(t2, target_orientation)             
    flair = ants.reorient_image2(flair, target_orientation)
    mask = ants.reorient_image2(mask, target_orientation)

    # 3. Co-register other modalities to T1 (Rigid 6-DoF - zero distortion)
    print("Running intra-subject rigid registration...")
    flair_to_t1 = ants.registration(fixed=t1, moving=flair, type_of_transform='Rigid')
    t2_to_t1 = ants.registration(fixed=t1, moving=t2, type_of_transform='Rigid') 

    # 4. Register the structural anchor (T1) to MNI152 (Affine - global scale/rotation, no warping)
    print("Running T1 to MNI affine registration...")
    t1_to_mni = ants.registration(fixed=mni, moving=t1, type_of_transform='Affine')

    # 5. Apply combined transforms to MNI space
    print("Applying combined transforms to MNI space...")
    
    # T1 only needs the T1-to-MNI affine matrix
    t1_mni = t1_to_mni['warpedmovout']
    
    # FLAIR needs: FLAIR -> T1 matrix AND THEN T1 -> MNI matrix
    flair_mni = ants.apply_transforms(
        fixed=mni, 
        moving=flair, 
        transformlist=[t1_to_mni['fwdtransforms'][0], flair_to_t1['fwdtransforms'][0]]
    )

    # T2 needs: T2 -> T1 matrix AND THEN T1 -> MNI matrix
    t2_mni = ants.apply_transforms(
        fixed=mni, 
        moving=t2, 
        transformlist=[t1_to_mni['fwdtransforms'][0], t2_to_t1['fwdtransforms'][0]]
    ) 

    # MASK needs: MASK -> T1 matrix AND THEN T1 -> MNI matrix
    # Use genericLabel (nearest neighbor) to preserve binary mask integer values
    if dataset_name == "miccai":
        mask_mni = ants.apply_transforms(
            fixed=mni, 
            moving=mask, 
            transformlist=[t1_to_mni['fwdtransforms'][0], flair_to_t1['fwdtransforms'][0]], 
            interpolator='genericLabel'
        )
    else:
        mask_mni = mask

    # Ensure output directory exist before saving
    os.makedirs(os.path.dirname(t1_output), exist_ok=True)

    ants.image_write(t1_mni, t1_output)
    ants.image_write(t2_mni, t2_output)
    ants.image_write(flair_mni, flair_output)
    ants.image_write(mask_mni, mask_output)
    print(f"Registration Complete for subject {subject_name}")
    print(f"Saved samples to {output_base}/{subject_name}")

    return {
        "t1": t1_output,
        "t2": t2_output,
        "flair": flair_output,
        "mask": mask_output
    }

def extract_brain(input_paths):
    t1_input = input_paths["t1"]
    t2_input = input_paths["t2"]
    flair_input = input_paths["flair"]
    
    t1_output = t1_input
    t2_output = t2_input
    flair_output = flair_input

    os.system(f"bet {t1_input} {t1_output}")
    os.system(f"bet {t2_input} {t2_output}")
    os.system(f"bet {flair_input} {flair_output}")
    
    print("Brain Extraction complete!")
    print(f"Saved samples to {os.path.dirname(t1_input)}")
    return {
        "t1": t1_output,
        "t2": t2_output,
        "flair": flair_output,
        "mask": input_paths["mask"]
    }

def n4bias_correction(input_path):

    """
    Performs N4 Bias Field Correction on T1, T2, and FLAIR skull-stripped images.
    
    Parameters:
        subject_dir (str): Path to the folder containing raw T1, T2, FLAIR files.
        output_dir (str): Path where corrected images will be saved.
    """
    # Define your specific file mappings (modify keys/extensions to match your file naming)
    modalities = {
        'T1': input_path["t1"],
        'T2': input_path["t2"],
        'FLAIR': input_path["flair"]
    }

    print("Starting N4 Bias Field Correction...")
    
    # Wrap the dictionary in tqdm for a clean progress bar
    for mod_name, file_name in tqdm(modalities.items(), desc="Processing Modalities"):
        input_path = file_name
        output_path = file_name
        
        # Verify the file actually exist before processing
        if not os.path.exists(input_path):
            print(f"\nWarning: {file_name} not found. Skipping {mod_name}.")
            continue
            
        # 1. Load the skull-stripped image
        img = ants.image_read(input_path)
        
        # 2. Auto-generate mask from non-zero voxels (ideal for skull-stripped data)
        # cleanup=0 ensures the exact boundaries of your stripped brain are preserved
        auto_mask = ants.get_mask(img, cleanup=0)
        
        # 3. Execute N4 Bias Field Correction
        # shrink_factor=2 balances speed and sub-voxel accuracy for clean data
        corrected_img = ants.n4_bias_field_correction(
            img,
            mask=auto_mask,
            shrink_factor=2,
            convergence={'iters': [50, 50, 50, 50], 'tol': 1e-07}
        )
        
        # 4. Save the corrected image
        ants.image_write(corrected_img, output_path)

    print(f"N4 bias correction complete! Saved samples to {os.path.dirname(modalities['T1'])}")
    output_path = {}
    for mode in modalities:
        output_path[mode.lower()] = modalities[mode]
    if "mask" in input_path:
        output_path["mask"] = input_path["mask"]
    return output_path

def z_score_normalize(input_path):
    t1_path = input_path["t1"]
    t2_path = input_path["t2"]
    flair_path = input_path["flair"]

    t1_img = nib.load(t1_path)
    t1_data = t1_img.get_fdata()

    t2_img = nib.load(t2_path)
    t2_data = t2_img.get_fdata()

    flair_img = nib.load(flair_path)
    flair_data = flair_img.get_fdata()

    t1_brain_mask = t1_data > 0.1
    t2_brain_mask = t2_data > 0.1
    flair_brain_mask = flair_data > 0.1

    t1_mean_val = np.mean(t1_data[t1_brain_mask])
    t1_std_val = np.std(t1_data[t1_brain_mask])

    t2_mean_val = np.mean(t2_data[t2_brain_mask])
    t2_std_val = np.std(t2_data[t2_brain_mask])

    flair_mean_val = np.mean(flair_data[flair_brain_mask])
    flair_std_val = np.std(flair_data[flair_brain_mask])

    t1_normalized = np.zeros_like(t1_data)
    t1_normalized[t1_brain_mask] = (t1_data[t1_brain_mask]-t1_mean_val)/(t1_std_val if t1_std_val > 0 else 1e-8)

    t2_normalized = np.zeros_like(t2_data)
    t2_normalized[t2_brain_mask] = (t2_data[t2_brain_mask]-t2_mean_val)/(t2_std_val if t2_std_val > 0 else 1e-8)

    flair_normalized = np.zeros_like(flair_data)
    flair_normalized[flair_brain_mask] = (flair_data[flair_brain_mask]-flair_mean_val)/(flair_std_val if flair_std_val > 0 else 1e-8)

    t1_normalized_img = nib.Nifti1Image(t1_normalized, t1_img.affine, t1_img.header)
    nib.save(t1_normalized_img, t1_path)

    t2_normalized_img = nib.Nifti1Image(t2_normalized, t2_img.affine, t2_img.header)
    nib.save(t2_normalized_img, t2_path)

    flair_normalized_img = nib.Nifti1Image(flair_normalized, flair_img.affine, flair_img.header)
    nib.save(flair_normalized_img, flair_path)

    print(f"Z-score normalization complete! Saved samples to {os.path.dirname(t1_path)}")
    return input_path

def resample_to_256(paths_dict, target_shape=(256, 256, 182)):
    """Resamples T1, T2, FLAIR, and MASK volumes to 256x256 in-plane."""
    print("Resampling volumes to 256x256...")
    for modality, path in paths_dict.items():
        if not os.path.exists(path):
            continue
        img = ants.image_read(path)
        is_mask = (modality == "mask")
        interp = 1 if is_mask else 0
        
        # Keep depth the same as the registered template space (182)
        new_shape = (target_shape[0], target_shape[1], img.shape[2])
        
        resampled = ants.resample_image(img, new_shape, use_voxels=True, interp_type=interp)
        ants.image_write(resampled, path)
    print("Resampling complete!")

def main():
    TEMPLATE_PATH = "/home/darshan/MS/preprocessing/TEMPLATE/MNI152_T1_1mm.nii.gz"
    samples = "/home/darshan/MS/preprocessing/samples.json"
    with open(samples, 'r') as f:
        data = json.load(f)

    for subject_name in data:
        print(f"Starting Preprocessing for subject {subject_name}")
        registered_paths = register(subject_name, data[subject_name], TEMPLATE_PATH, "/home/darshan/MS/data/PREPROCESSED256")
        brain_extracted_paths = extract_brain(registered_paths)
        n4_paths = n4bias_correction(brain_extracted_paths)
        z_score_paths = z_score_normalize(n4_paths)
        resample_to_256(z_score_paths)
        print(f"Preprocessing complete for subject {subject_name}!")
        print(f"Images saved to {os.path.dirname(z_score_paths['t1'])}\n")

    print("Preprocessing complete for all patients")

if __name__ == "__main__":
    main()
