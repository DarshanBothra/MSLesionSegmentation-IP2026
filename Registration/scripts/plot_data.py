import os
import sys
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Plots raw and preprocessed data

index = {
    "ISBI": {
        "FLAIR": "/Users/darshan/Code/Registration/data/ISBI/orig/training04_03_flair.nii.gz",
        "T1": "/Users/darshan/Code/Registration/data/ISBI/orig/training04_03_mprage.nii.gz",
        "T2": "/Users/darshan/Code/Registration/data/ISBI/orig/training04_03_t2.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/data/ISBI/masks/training04_03_mask1.nii"
    },
    "MICCAI": {
        "FLAIR": "/Users/darshan/Code/Registration/data/MICCAI/Raw_Data/FLAIR.nii.gz",
        "T1": "/Users/darshan/Code/Registration/data/MICCAI/Raw_Data/T1.nii.gz",
        "T2": "/Users/darshan/Code/Registration/data/MICCAI/Raw_Data/T2.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/data/MICCAI/Masks/Consensus.nii.gz",
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
        "FLAIR": "/Users/darshan/Code/Registration/data/ISBI/preprocessed/training04_03_flair_pp.nii",
        "T1": "/Users/darshan/Code/Registration/data/ISBI/preprocessed/training04_03_mprage_pp.nii",
        "T2": "/Users/darshan/Code/Registration/data/ISBI/preprocessed/training04_03_t2_pp.nii",
        "MASK": "/Users/darshan/Code/Registration/data/ISBI/masks/training04_03_mask1.nii"
    },
    "MICCAI": {
        "FLAIR": "/Users/darshan/Code/Registration/data/MICCAI/Preprocessed_Data/FLAIR_preprocessed.nii.gz",
        "T1": "/Users/darshan/Code/Registration/data/MICCAI/Preprocessed_Data/T1_preprocessed.nii.gz",
        "T2": "/Users/darshan/Code/Registration/data/MICCAI/Preprocessed_Data/T2_preprocessed.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/data/MICCAI/Masks/Consensus.nii.gz",
    },
    "MSLegSeg": {
        "FLAIR": "/Users/darshan/Code/Registration/data/MSLegSeg/Preprocessed/P6_T2_FLAIR.nii.gz",
        "T1": "/Users/darshan/Code/Registration/data/MSLegSeg/Preprocessed/P6_T2_T1.nii.gz",
        "T2": "/Users/darshan/Code/Registration/data/MSLegSeg/Preprocessed/P6_T2_T2.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/data/MSLegSeg/Raw/P6_T2_MASK.nii.gz",
    }
}


def load_nifti_image(path):
    """Loads a NIfTI image, squeezes any redundant dimensions, and returns (data, voxel_size_str)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    img = nib.load(path)
    if nib.aff2axcodes(img.affine) != ('L', 'A', 'S'):
        img = img.as_reoriented(nib.orientations.axcodes2ornt(('L', 'A', 'S')))
    data = np.squeeze(img.get_fdata())
    zooms = img.header.get_zooms()
    voxel_size_str = f"{zooms[0]:.2f} x {zooms[1]:.2f} x {zooms[2]:.2f} mm"
    return data, voxel_size_str


def create_preprocessed_plot(dataset):
    """Creates the figure and subplots for preprocessed images, synced to a single Z-slice slider."""
    print(f"Loading preprocessed images for {dataset}...")
    paths = index_preprocessed[dataset]
    
    data_dict = {}
    voxel_sizes = {}
    for modality in ["FLAIR", "T1", "T2", "MASK"]:
        try:
            data_dict[modality], voxel_sizes[modality] = load_nifti_image(paths[modality])
        except Exception as e:
            print(f"Error loading preprocessed {modality}: {e}")
            data_dict[modality] = None
            voxel_sizes[modality] = "N/A"

    # Get the minimum Z dimension from the successfully loaded images
    z_dims = [data_dict[mod].shape[2] for mod in data_dict if data_dict[mod] is not None]
    if not z_dims:
        print(f"Error: No preprocessed images could be loaded for {dataset}.")
        return None, None
        
    min_z = min(z_dims)
    init_slice = min_z // 2

    # Set up figure and 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    fig.canvas.manager.set_window_title(f"Preprocessed Images - {dataset}")
    fig.suptitle(f"Preprocessed Images: {dataset} (Synced Z-Slice)", fontsize=14, fontweight='bold')
    
    modalities = [
        ("FLAIR", axes[0, 0]),
        ("T1", axes[0, 1]),
        ("T2", axes[1, 0]),
        ("MASK", axes[1, 1])
    ]
    
    im_objs = {}
    for mod, ax in modalities:
        data = data_dict[mod]
        if data is None:
            # Display elegant text message in this subplot
            ax.set_facecolor('#1a1a1a')
            ax.text(0.5, 0.5, f"Error Loading\n{mod}\n(File Damaged)", 
                    color='#ff5555', fontsize=10, fontweight='bold', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
            
        # Transpose (.T) and origin='lower' to align with standard viewing
        im = ax.imshow(data[:, :, init_slice].T, cmap='gray', origin='lower')
        ax.set_title(f"{mod} (Slice {init_slice}/{min_z - 1})\nSize: {voxel_sizes[mod]}", fontsize=10, fontweight='semibold')
        ax.axis('off')
        im_objs[mod] = im

    # Adjust layout to make room for the single slider at the bottom
    plt.subplots_adjust(bottom=0.15, top=0.88, hspace=0.3, wspace=0.3)

    # Single slider at the bottom
    slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax=slider_ax,
        label='Z-Slice ',
        valmin=0,
        valmax=min_z - 1,
        valinit=init_slice,
        valfmt='%i'
    )

    def update(val):
        slice_idx = int(slider.val)
        for mod, ax in modalities:
            data = data_dict[mod]
            if data is not None:
                im_objs[mod].set_data(data[:, :, slice_idx].T)
                ax.set_title(f"{mod} (Slice {slice_idx}/{min_z - 1})\nSize: {voxel_sizes[mod]}", fontsize=10, fontweight='semibold')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    slider_ax._slider = slider  # Keep strong reference to prevent GC

    return fig, slider


def create_raw_plot(dataset):
    """Creates the figure and subplots for raw images, each controlled by an independent Z-slice slider."""
    print(f"Loading raw images for {dataset}...")
    paths = index[dataset]
    
    data_dict = {}
    voxel_sizes = {}
    for modality in ["FLAIR", "T1", "T2", "MASK"]:
        try:
            data_dict[modality], voxel_sizes[modality] = load_nifti_image(paths[modality])
        except Exception as e:
            print(f"Error loading raw {modality}: {e}")
            data_dict[modality] = None
            voxel_sizes[modality] = "N/A"

    # Check if we successfully loaded at least one raw image
    loaded_any = any(data_dict[mod] is not None for mod in data_dict)
    if not loaded_any:
        print(f"Error: No raw images could be loaded for {dataset}.")
        return None, None

    # Set up figure and 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    fig.canvas.manager.set_window_title(f"Raw Images - {dataset}")
    fig.suptitle(f"Raw Images: {dataset} (Independent Z-Slices)", fontsize=14, fontweight='bold')
    
    modalities = [
        ("FLAIR", axes[0, 0]),
        ("T1", axes[0, 1]),
        ("T2", axes[1, 0]),
        ("MASK", axes[1, 1])
    ]
    
    sliders = []

    # Extra vertical spacing to allow sliders beneath each subplot
    plt.subplots_adjust(bottom=0.1, top=0.88, hspace=0.45, wspace=0.3)

    for mod, ax in modalities:
        data = data_dict[mod]
        if data is None:
            # Display elegant text message in this subplot
            ax.set_facecolor('#1a1a1a')
            ax.text(0.5, 0.5, f"Error Loading\n{mod}\n(File Damaged)", 
                    color='#ff5555', fontsize=10, fontweight='bold', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue
            
        z_dim = data.shape[2]
        init_slice = z_dim // 2
        
        im = ax.imshow(data[:, :, init_slice].T, cmap='gray', origin='lower')
        ax.set_title(f"{mod} (Slice {init_slice}/{z_dim - 1})\nSize: {voxel_sizes[mod]}", fontsize=10, fontweight='semibold')
        ax.axis('off')
        
        # Position slider directly below this specific axes
        pos = ax.get_position()
        slider_ax = fig.add_axes([pos.x0, pos.y0 - 0.06, pos.width, 0.02])
        
        slider = Slider(
            ax=slider_ax,
            label='Z-Slice ',
            valmin=0,
            valmax=z_dim - 1,
            valinit=init_slice,
            valfmt='%i'
        )
        
        # Closure to bind current loop variables
        def make_update(im_obj, ax_obj, mod_name, data_arr, max_val, s_obj, size_str):
            def update(val):
                slice_idx = int(s_obj.val)
                im_obj.set_data(data_arr[:, :, slice_idx].T)
                ax_obj.set_title(f"{mod_name} (Slice {slice_idx}/{max_val - 1})\nSize: {size_str}", fontsize=10, fontweight='semibold')
                fig.canvas.draw_idle()
            return update
            
        slider.on_changed(make_update(im, ax, mod, data, z_dim, slider, voxel_sizes[mod]))
        slider_ax._slider = slider  # Keep strong reference
        sliders.append(slider)

    return fig, sliders


def main():
    parser = argparse.ArgumentParser(description="Plot raw and preprocessed NIfTI images with interactive sliders.")
    parser.add_argument(
        "dataset", 
        nargs="?", 
        choices=["ISBI", "MICCAI", "MSLegSeg"], 
        help="Name of the dataset (ISBI, MICCAI, or MSLegSeg)"
    )
    args = parser.parse_args()

    dataset = args.dataset
    if not dataset:
        print("Available datasets: ISBI, MICCAI, MSLegSeg")
        dataset = input("Please enter the dataset name: ").strip()
        # Case-insensitive autocomplete/correction
        matched = [d for d in ["ISBI", "MICCAI", "MSLegSeg"] if d.lower() == dataset.lower()]
        if matched:
            dataset = matched[0]
        else:
            print(f"Error: Invalid dataset '{dataset}'. Must be one of: ISBI, MICCAI, MSLegSeg.")
            sys.exit(1)

    # 1. Create the preprocessed plot
    fig_pre, slider_pre = create_preprocessed_plot(dataset)

    # 2. Create the raw plot
    fig_raw, sliders_raw = create_raw_plot(dataset)

    # 3. Check if we have at least one valid figure to show
    if fig_pre is None and fig_raw is None:
        print("Error: Both preprocessed and raw plots failed to load. Exiting.")
        sys.exit(1)
    elif fig_pre is None:
        print("Warning: Preprocessed plot failed to load, but raw plot succeeded. Showing raw plot only.")
    elif fig_raw is None:
        print("Warning: Raw plot failed to load, but preprocessed plot succeeded. Showing preprocessed plot only.")

    # 4. Open active windows simultaneously on the main thread
    print("Opening plot windows. Close them to exit.")
    plt.show()


if __name__ == "__main__":
    main()


# NOTES

'''
- Consensus is empty in MICCAI dataset! Take brain mask of one expert!

- 

'''