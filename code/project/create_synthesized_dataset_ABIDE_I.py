import sys
import os

import SimpleITK as sitk
import disptools.displacements as dsp
import numpy as np
from matplotlib import pyplot as plt

from utils.utils import create_2D_image, get_ABIDE_I_subject, get_ABIDE_II_subject
from utils.const import ABIDE_I_MNI, ABIDE_II_MNI


if __name__ == "__main__":
    samples_ABIDE_I, n_subjects_lower_21_I = get_ABIDE_I_subject()
    samples_ABIDE_II, n_subjects_lower_21_II = get_ABIDE_II_subject()

    print(f"Number of subjects in ABIDE I: {len(samples_ABIDE_I)}")
    print(f"Number of subjects in ABIDE II: {len(samples_ABIDE_II)}")
    print(f"Number of subjects with age <= 21 in ABIDE I: {n_subjects_lower_21_I}")
    print(f"Number of subjects with age <= 21 in ABIDE II: {n_subjects_lower_21_II}")

    atlas = sitk.ReadImage(
        "/projectnb/ace-genetics/jueqiw/dataset/MRI_template/Atlas/BN_Atlas_246_space-MNI152NLin2009cAsym.nii.gz"
    )

    rng = np.random.default_rng(seed=42)
    all_labels = np.arange(1, 247)  # 246 parcels
    n_rois = 10
    chosen = rng.choice(all_labels, n_rois, replace=False)
    atrophy_labels = chosen[: n_rois // 2]
    expand_labels = chosen[n_rois // 2 :]

    # give a random atrophy factor and expansion factor for each label
    # atrophy factor is between 0.8 and 1
    # expansion factor is between 1 and 1.2
    atrophy_factor = rng.uniform(0.8, 1.0, size=n_rois // 2)
    expand_factor = rng.uniform(1.0, 1.2, size=n_rois // 2)
    lab_np = sitk.GetArrayFromImage(atlas)
    jac_np = np.ones_like(lab_np, dtype=np.float32)

    for roi, factor in zip(atrophy_labels, atrophy_factor):
        jac_np[lab_np == roi] = factor
    for roi, factor in zip(expand_labels, expand_factor):
        jac_np[lab_np == roi] = factor

    mask_np = (jac_np != 1.0).astype(np.uint8)
    jacobian_img = sitk.GetImageFromArray(jac_np)
    jacobian_img.CopyInformation(atlas)
    mask_img = sitk.GetImageFromArray(mask_np)
    mask_img.CopyInformation(atlas)

    disp = dsp.displacement(
        jacobian_img,
        mask=mask_img,
        levels=1,
        epsilon=1e-2,
        it_max=5000,
        algorithm="greedy",
    )

    sitk.WriteImage(disp, "sim_BN_dispfield_new.nii.gz")
    sitk.WriteImage(jacobian_img, "sim_BN_targetJac_new.nii.gz")

    img = sitk.ReadImage(
        "/projectnb/ace-ig/ABIDE/ABIDE_I_BIDS/derivatives/MNI/sub-51606/anat/sub-51606_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
    )
    img = sitk.Resample(
        img, atlas, sitk.Transform(), sitk.sitkLinear, 0.0, img.GetPixelIDValue()
    )
    img = sitk.Resample(
        img,
        atlas,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0.0,
        img.GetPixelIDValue(),
    )

    disp_cast = sitk.Cast(disp, sitk.sitkVectorFloat64)
    img_disp = sitk.DisplacementFieldTransform(disp_cast)
    img_disp = sitk.Resample(img, img_disp, sitk.sitkLinear, 0.0, img.GetPixelIDValue())
    sitk.WriteImage(img_disp, "sim_BN_atrophy_new.nii.gz")
    sitk.WriteImage(mask_img, "sim_BN_mask_new.nii.gz")

    print("Finished!  →  sim_BN_dispfield.nii.gz")
    # count how long it takes to run this script
    end_time = os.times()[0]
    print(f"Total time taken: {end_time - start_time} seconds")

    # # load the images for visualization
    # img = sitk.ReadImage("/projectnb/ace-ig/ABIDE/ABIDE_I_BIDS/derivatives/MNI/sub-51606/anat/sub-51606_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz")
    # img_disp = sitk.ReadImage("sim_BN_atrophy.nii.gz")  # This is the transformed brain image
    # disp_field = sitk.ReadImage("sim_BN_dispfield.nii.gz")  # This is the displacement field (3D vector)
    # jacobian_img = sitk.ReadImage("sim_BN_targetJac.nii.gz")  # This is the target jacobian

    # # Convert SimpleITK images to numpy arrays for plotting
    # img_np = sitk.GetArrayFromImage(img)
    # img_disp_np = sitk.GetArrayFromImage(img_disp)
    # disp_field_np = sitk.GetArrayFromImage(disp_field)

    # print(f"Original image shape: {img_np.shape}")
    # print(f"Transformed image shape: {img_disp_np.shape}")
    # print(f"Displacement field shape: {disp_field_np.shape}")
    # print(f"Displacement field data type: {disp_field_np.dtype}")

    # # Get middle slice
    # mid_slice = img_np.shape[0] // 2

    # # difference of the two images
    # diff_np = img_disp_np - img_np

    # # Get the jacobian image for overlay
    # jac_display = sitk.GetArrayFromImage(jacobian_img)

    # # Create a figure with subplots
    # plt.figure(figsize=(20, 5))

    # plt.subplot(1, 5, 1)
    # plt.imshow(img_np[mid_slice, :, :], cmap='gray')
    # plt.title("Original Image")
    # plt.axis('off')

    # plt.subplot(1, 5, 2)
    # plt.imshow(img_disp_np[mid_slice, :, :], cmap='gray')
    # plt.title("Transformed Image")
    # plt.axis('off')

    # plt.subplot(1, 5, 3)
    # plt.imshow(diff_np[mid_slice, :, :], cmap='RdBu_r')
    # plt.title("Difference Image")
    # plt.axis('off')
    # plt.colorbar()

    # plt.subplot(1, 5, 4)
    # # Display displacement field magnitude
    # if len(disp_field_np.shape) == 4:
    #     # 4D displacement field: (z, y, x, components)
    #     disp_magnitude = np.sqrt(np.sum(disp_field_np[mid_slice, :, :, :]**2, axis=-1))
    # else:
    #     # Handle other cases
    #     disp_magnitude = np.abs(disp_field_np[mid_slice, :, :])

    # plt.imshow(disp_magnitude, cmap='hot')
    # plt.title("Displacement Magnitude")
    # plt.axis('off')
    # plt.colorbar()

    # plt.subplot(1, 5, 5)
    # # Draw jacobian map over original image
    # plt.imshow(img_np[mid_slice, :, :], cmap='gray')
    # # Create a mask for non-unity jacobian values
    # jac_slice = jac_display[mid_slice, :, :]
    # jac_mask = (jac_slice != 1.0)
    # # Overlay jacobian map only where it's different from 1
    # plt.imshow(np.ma.masked_where(~jac_mask, jac_slice),
    #           cmap='RdBu_r', alpha=0.6, vmin=0.8, vmax=1.2)
    # plt.title("Jacobian Overlay")
    # plt.axis('off')
    # plt.colorbar(label='Jacobian Value')

    # plt.tight_layout()
    # plt.savefig("brain_analysis_visualization.png", dpi=150, bbox_inches='tight')
    # plt.show()

    # print("Visualization completed!")
