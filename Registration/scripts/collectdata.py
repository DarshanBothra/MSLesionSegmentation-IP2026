# ISBI - patient training04 tp 01
# MICCAI - Center 07 patient 05     
# MSLegSeg - Patient P6 tp T2
import os
import ants

def register(t1_path, t2_path, flair_path, mask_path, template_path, t1_output, t2_output, flair_output, mask_output):
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
    mask_mni = ants.apply_transforms(
        fixed=mni, 
        moving=mask, 
        transformlist=[t1_to_mni['fwdtransforms'][0], flair_to_t1['fwdtransforms'][0]], 
        interpolator='genericLabel'
    )

    print("T1 Orientation:", t1_mni.orientation)
    print("T2 Orientation:", t2_mni.orientation)
    print("FLAIR Orientation:", flair_mni.orientation)
    print("MASK Orientation:", mask_mni.orientation)

    # Ensure output directory exists before saving
    os.makedirs(os.path.dirname(t1_output), exist_ok=True)

    ants.image_write(t1_mni, t1_output)
    ants.image_write(t2_mni, t2_output)
    ants.image_write(flair_mni, flair_output)
    ants.image_write(mask_mni, mask_output)


TEMPLATE_PATH = "/Users/darshan/Code/Registration/data/MNI152_T1_1mm.nii.gz"
index = {
    "ISBI": {
        "FLAIR": "/Users/darshan/Code/Registration/data/ISBI/orig/training04_01_flair.nii.gz",
        "T1": "/Users/darshan/Code/Registration/data/ISBI/orig/training04_01_mprage.nii.gz",
        "T2": "/Users/darshan/Code/Registration/data/ISBI/orig/training04_01_t2.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/data/ISBI/masks/training04_01_mask1.nii"
    },
    "MICCAI": {
        "FLAIR": "/Users/darshan/Code/Registration/data/MICCAI/Raw_Data/FLAIR.nii.gz",
        "T1": "/Users/darshan/Code/Registration/data/MICCAI/Raw_Data/T1.nii.gz",
        "T2": "/Users/darshan/Code/Registration/data/MICCAI/Raw_Data/T2.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/data/MICCAI/Masks/ManualSegmentation_1.nii.gz",
    },
    "MSLegSeg": {
        "FLAIR": "/Users/darshan/Code/Registration/data/MSLegSeg/Raw/P6_T2_FLAIR.nii.gz",
        "T1": "/Users/darshan/Code/Registration/data/MSLegSeg/Raw/P6_T2_T1.nii.gz",
        "T2": "/Users/darshan/Code/Registration/data/MSLegSeg/Raw/P6_T2_T2.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/data/MSLegSeg/Raw/P6_T2_MASK.nii.gz",
    }
}

index_preprocessed = {
    "ISBI": {
        "FLAIR": "/Users/darshan/Code/Registration/data/ISBI/preprocessed/training04_01_flair.nii.gz",
        "T1": "/Users/darshan/Code/Registration/data/ISBI/preprocessed/training04_01_mprage.nii.gz",
        "T2": "/Users/darshan/Code/Registration/data/ISBI/preprocessed/training04_01_t2.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/data/ISBI/masks/training04_01_mask1.nii"
    },
    "MICCAI": {
        "FLAIR": "/Users/darshan/Code/Registration/data/MICCAI/Preprocessed_Data/FLAIR_preprocessed.nii.gz",
        "T1": "/Users/darshan/Code/Registration/data/MICCAI/Preprocessed_Data/T1_preprocessed.nii.gz",
        "T2": "/Users/darshan/Code/Registration/data/MICCAI/Preprocessed_Data/T2_preprocessed.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/data/MICCAI/Masks/ManualSegmentation_1.nii.gz",
    },
    "MSLegSeg": {
        "FLAIR": "/Users/darshan/Code/Registration/data/MSLegSeg/Preprocessed/P6_T2_FLAIR.nii.gz",
        "T1": "/Users/darshan/Code/Registration/data/MSLegSeg/Preprocessed/P6_T2_T1.nii.gz",
        "T2": "/Users/darshan/Code/Registration/data/MSLegSeg/Preprocessed/P6_T2_T2.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/data/MSLegSeg/Raw/P6_T2_MASK.nii.gz",
    }
}


# Registration Script

for dataset in index:
    output_path = os.path.join("/Users/darshan/Code/Registration/output", dataset)
    print(f"--- Running ANTsPy Registration for {dataset} ---")
    
    # We try-catch each dataset individually so that a corrupted input on one doesn't stop others from processing!
    try:
        register(
            index[dataset]["T1"], 
            index[dataset]["T2"], 
            index[dataset]["FLAIR"], 
            index[dataset]["MASK"], 
            TEMPLATE_PATH, 
            os.path.join(output_path, "t1.nii.gz"), 
            os.path.join(output_path, "t2.nii.gz"), 
            os.path.join(output_path, "flair.nii.gz"), 
            os.path.join(output_path, "mask.nii.gz")
        )
        print("Saved successfully to:", output_path)
    except Exception as e:
        print(f"Error registering {dataset}: {e}")
    print()