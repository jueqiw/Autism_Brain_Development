import sys
import os

import SimpleITK as sitk
import disptools.displacements as dsp
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # draw the plt of the mask
    # load the nii.gz file
    # img = sitk.ReadImage(
    #     "/projectnb/ace-genetics/ABIDE/ABIDE_I/sub-51328/sim_BN_ta.nii.gz"
    # )
    # img_np = sitk.GetArrayFromImage(img)
    # plt.imshow(img_np[img_np.shape[0] // 2, :, :], cmap="gray")
    # plt.title("Mask Image")
    # plt.axis("off")
    # plt.savefig("mask_image.png", dpi=150, bbox_inches="tight")
    # plt.show()
    # # compute how long it takes to run this script
    # start_time = os.times()[0]
    # # load the Brainnetome atlas
    # atlas = sitk.ReadImage(
    #     "/projectnb/ace-genetics/jueqiw/dataset/MRI_template/Atlas/BN_Atlas_246_space-MNI152NLin2009cAsym.nii.gz"
    # )

    # load the images for visualization
    img = sitk.ReadImage(
        "/projectnb/ace-ig/ABIDE/ABIDE_I_BIDS/derivatives/MNI/sub-51328/anat/sub-51328_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
    )
    img_disp = sitk.ReadImage(
        "/projectnb/ace-genetics/ABIDE/ABIDE_I/sub-51328/transformed_0.nii.gz"
    )  # This is the transformed brain image
    jacobian_img = sitk.ReadImage(
        "/projectnb/ace-genetics/ABIDE/ABIDE_I/sub-51328/sim_BN_targetJac_0.nii.gz"
    )  # This is the target jacobian
    disp_field = sitk.ReadImage(
        "/projectnb/ace-genetics/ABIDE/ABIDE_I/sub-51328/sim_BN_dispfield_0.nii.gz"
    )  # This is the displacement field (3D vector)

    # Convert SimpleITK images to numpy arrays for plotting
    img_np = sitk.GetArrayFromImage(img)
    img_disp_np = sitk.GetArrayFromImage(img_disp)
    disp_field_np = sitk.GetArrayFromImage(disp_field)

    # Get middle slice
    mid_slice = img_np.shape[0] // 2

    # difference of the two images
    diff_np = img_disp_np - img_np

    # Get the jacobian image for overlay
    jac_display = sitk.GetArrayFromImage(jacobian_img)

    # Create a figure with subplots
    plt.figure(figsize=(25, 5))

    # flip the images for correct orientation

    plt.subplot(1, 6, 1)
    plt.imshow(np.flip(img_np[mid_slice, :, :], axis=0), cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 6, 2)
    plt.imshow(np.flip(img_disp_np[mid_slice, :, :], axis=0), cmap="gray")
    plt.title("Transformed Image")
    plt.axis("off")

    plt.subplot(1, 6, 3)
    plt.imshow(np.flip(diff_np[mid_slice, :, :], axis=0), cmap="RdBu_r")
    plt.title("Difference Image")
    plt.axis("off")
    plt.colorbar()

    plt.subplot(1, 6, 4)
    # Display displacement field magnitude
    if len(disp_field_np.shape) == 4:
        # 4D displacement field: (z, y, x, components)
        disp_magnitude = np.sqrt(
            np.sum(disp_field_np[mid_slice, :, :, :] ** 2, axis=-1)
        )
    else:
        # Handle other cases
        disp_magnitude = np.abs(disp_field_np[mid_slice, :, :])

    plt.imshow(np.flip(disp_magnitude, axis=0), cmap="hot")
    plt.title("Displacement Magnitude")
    plt.axis("off")
    plt.colorbar()

    plt.subplot(1, 6, 5)
    # Draw jacobian map over original image
    plt.imshow(np.flip(img_np[mid_slice, :, :], axis=0), cmap="gray")
    # Create a mask for non-unity jacobian values
    jac_slice = jac_display[mid_slice, :, :]
    jac_mask = jac_slice != 1.0
    # Overlay jacobian map only where it's different from 1
    # Make it more visible by increasing alpha and adjusting contrast
    jac_overlay = np.ma.masked_where(~jac_mask, jac_slice)
    plt.imshow(
        np.flip(jac_overlay, axis=0), cmap="RdBu_r", alpha=0.8, vmin=0.75, vmax=1.25
    )
    plt.title("Jacobian Overlay on Original")
    plt.axis("off")
    plt.colorbar(label="Jacobian Value")

    plt.subplot(1, 6, 6)
    # Draw jacobian map over transformed image
    plt.imshow(np.flip(img_disp_np[mid_slice, :, :], axis=0), cmap="gray")
    # Create a mask for non-unity jacobian values
    jac_mask_disp = jac_slice != 1.0
    # Overlay jacobian map only where it's different from 1
    # Make it more visible by increasing alpha and adjusting contrast
    jac_overlay_disp = np.ma.masked_where(~jac_mask_disp, jac_slice)
    plt.imshow(
        np.flip(jac_overlay_disp, axis=0),
        cmap="RdBu_r",
        alpha=0.8,
        vmin=0.75,
        vmax=1.25,
    )
    plt.title("Jacobian Overlay on Transformed")
    plt.axis("off")
    plt.colorbar(label="Jacobian Value")

    plt.tight_layout()
    plt.savefig("brain_analysis_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Visualization completed!")
