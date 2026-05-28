import os
import sys
import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Plots original preprocessed data and newly registered output data

index = {
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

index_preprocessed = {
    "ISBI": {
        "FLAIR": "/Users/darshan/Code/Registration/output/ISBI/flair.nii.gz",
        "T1": "/Users/darshan/Code/Registration/output/ISBI/t1.nii.gz",
        "T2": "/Users/darshan/Code/Registration/output/ISBI/t2.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/output/ISBI/mask.nii.gz"
    },
    "MICCAI": {
        "FLAIR": "/Users/darshan/Code/Registration/output/MICCAI/flair.nii.gz",
        "T1": "/Users/darshan/Code/Registration/output/MICCAI/t1.nii.gz",
        "T2": "/Users/darshan/Code/Registration/output/MICCAI/t2.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/output/MICCAI/mask.nii.gz",
    },
    "MSLegSeg": {
        "FLAIR": "/Users/darshan/Code/Registration/output/MSLegSeg/flair.nii.gz",
        "T1": "/Users/darshan/Code/Registration/output/MSLegSeg/t1.nii.gz",
        "T2": "/Users/darshan/Code/Registration/output/MSLegSeg/t2.nii.gz",
        "MASK": "/Users/darshan/Code/Registration/output/MSLegSeg/mask.nii.gz",
    }
}


def load_nifti_image(path):
    """Loads a NIfTI image, squeezes any redundant dimensions, aligns orientation, and returns (data, voxel_size_str)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    img = nib.load(path)
    if nib.aff2axcodes(img.affine) != ('L', 'A', 'S'):
        img = img.as_reoriented(nib.orientations.axcodes2ornt(('L', 'A', 'S')))
    data = np.squeeze(img.get_fdata())
    zooms = img.header.get_zooms()
    voxel_size_str = f"{zooms[0]:.2f} x {zooms[1]:.2f} x {zooms[2]:.2f} mm"
    return data, voxel_size_str


def create_synced_plot(dataset, is_preprocessed_output=False):
    """Creates a figure and subplots, synced to a single Z-slice slider, supporting per-subplot fallback."""
    if is_preprocessed_output:
        title_prefix = "Registered Output"
        paths = index_preprocessed[dataset]
        print(f"Loading registered output images for {dataset}...")
    else:
        title_prefix = "Original Preprocessed"
        paths = index[dataset]
        print(f"Loading original preprocessed images for {dataset}...")
        
    data_dict = {}
    voxel_sizes = {}
    for modality in ["FLAIR", "T1", "T2", "MASK"]:
        try:
            data_dict[modality], voxel_sizes[modality] = load_nifti_image(paths[modality])
        except Exception as e:
            print(f"Error loading {title_prefix.lower()} {modality}: {e}")
            data_dict[modality] = None
            voxel_sizes[modality] = "N/A"

    # Get the minimum Z dimension from successfully loaded images
    z_dims = [data_dict[mod].shape[2] for mod in data_dict if data_dict[mod] is not None]
    if not z_dims:
        print(f"Error: No {title_prefix.lower()} images could be loaded for {dataset}.")
        return None, None
        
    min_z = min(z_dims)
    init_slice = min_z // 2

    # Set up figure and 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    fig.canvas.manager.set_window_title(f"{title_prefix} Images - {dataset}")
    fig.suptitle(f"{title_prefix} Images: {dataset} (Synced Z-Slice)", fontsize=14, fontweight='bold')
    
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
            # Display elegant text error card in this subplot
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
    slider_ax._slider = slider  # Keep strong reference

    return fig, slider


def main():
    parser = argparse.ArgumentParser(description="Plot original preprocessed and newly registered output NIfTI images.")
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
        # Case-insensitive autocomplete
        matched = [d for d in ["ISBI", "MICCAI", "MSLegSeg"] if d.lower() == dataset.lower()]
        if matched:
            dataset = matched[0]
        else:
            print(f"Error: Invalid dataset '{dataset}'. Must be one of: ISBI, MICCAI, MSLegSeg.")
            sys.exit(1)

    # 1. Create the original preprocessed plot
    fig_orig, slider_orig = create_synced_plot(dataset, is_preprocessed_output=False)

    # 2. Create the registered output plot
    fig_reg, slider_reg = create_synced_plot(dataset, is_preprocessed_output=True)

    # 3. Check if we have at least one valid figure to show
    if fig_orig is None and fig_reg is None:
        print("Error: Both original preprocessed and registered output plots failed to load. Exiting.")
        sys.exit(1)
    elif fig_orig is None:
        print("Warning: Original preprocessed plot failed to load, but registered output plot succeeded. Showing registered only.")
    elif fig_reg is None:
        print("Warning: Registered output plot failed to load, but original preprocessed plot succeeded. Showing original only.")

    # 4. Open active windows simultaneously on the main thread
    print("Opening plot windows. Close them to exit.")
    plt.show()


if __name__ == "__main__":
    main()
