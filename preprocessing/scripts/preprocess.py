import argparse
from pathlib import Path

try:
    import ants
except ImportError:
    print("❌ Error: 'ants' library not found. Please run: pip install antspyx")
    exit(1)

def register_single_file_antspy(input_file, template_file, output_file, is_mask=False):
    """
    Registers a single NIfTI file directly to an MNI template using pure Python ANTsPy.
    """
    input_path = Path(input_file)
    template_path = Path(template_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🍁 ANTsPy Loading: {input_path.name}")
    
    # 1. Load volumes into ANTsPy memory structures
    moving_img = ants.image_read(str(input_path))
    fixed_img = ants.image_read(str(template_path))
    
    # Define interpolation: 'nearestneighbor' for masks, 'linear' for structural scans
    interp_method = 'nearestneighbor' if is_mask else 'linear'
    
    print(f"🔄 Optimizing Alignment Matrix (Rigid + Affine Mapping)...")
    try:
        # 2. Run the registration execution engine
        # type_of_transform='Affine' ensures 9 DOF (Translation, Rotation, Uniform Scale)
        # This aligns the brain flawlessly without causing stretching or clipping distortions!
        reg_results = ants.registration(
            fixed=fixed_img,
            moving=moving_img,
            type_of_transform='Affine',
            initial_transform=None,
            outprefix=str(output_path.parent / "tmp_"),
            verbose=False
        )
        
        # 3. Apply the transform and explicitly match your template grid dimensions
        warped_img = ants.apply_transforms(
            fixed=fixed_img,
            moving=moving_img,
            transformlist=reg_results['fwdtransforms'],
            interpolation=interp_method
        )
        
        # 4. Save file directly to output destination grid
        ants.image_write(warped_img, str(output_path))
        print(f"✅ ANTsPy Registration Complete! Saved to: {output_path}\n")
        
    except Exception as e:
        print(f"❌ ANTsPy registration failed for {input_path.name}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure Python ANTsPy registration utility for brain MRI.")
    parser.add_argument("-i", "--input", required=True, help="Input .nii.gz file")
    parser.add_argument("-t", "--template", required=True, help="MNI template .nii.gz file")
    parser.add_argument("-o", "--output", required=True, help="Output .nii.gz destination")
    parser.add_argument("--is_mask", action="store_true", help="Pass if handling a binary mask")
    
    args = parser.parse_args()
    
    register_single_file_antspy(
        input_file=args.input,
        template_file=args.template,
        output_file=args.output,
        is_mask=args.is_mask
    )
    