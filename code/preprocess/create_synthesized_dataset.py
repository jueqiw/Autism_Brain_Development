import sys
import os
import pickle

import SimpleITK as sitk
import disptools.displacements as dsp
import numpy as np
from matplotlib import pyplot as plt

from utils.utils import get_ABIDE_I_subject, get_ABIDE_II_subject, create_training_samples
from utils.const import ABIDE_I_MNI, ABIDE_II_MNI


if __name__ == "__main__":
    with open("samples_ABIDE_merged.pkl", "rb") as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} samples from the pickle file.")

    subjects = []
    n_subjects = 0
    for sample in samples:
        if sample["age"] <= 35:
            n_subjects += 1
            subjects.append(sample)

    print(f"Number of subjects with age <= 35: {n_subjects}")

    with open("subjects_ABIDE_lower_35.pkl", "wb") as f:
        pickle.dump(subjects, f)

    sys.exit(0)

    # create 20 jobs in total
    n_jobs = 20
    n_subjects = len(subjects)
    subjects_per_job = n_subjects // n_jobs
    for i in range(n_jobs):
        start_idx = i * subjects_per_job
        end_idx = (i + 1) * subjects_per_job if i < n_jobs - 1 else n_subjects
        job_subjects = subjects[start_idx:end_idx]

        # create a folder for each job
        job_dir = f"job_{i + 1}"
        os.makedirs(job_dir, exist_ok=True)

        # save the subjects for this job
        with open(os.path.join(job_dir, "subjects.pkl"), "wb") as f:
            pickle.dump(job_subjects, f)


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
    img = sitk.ReadImage("/projectnb/ace-ig/ABIDE/ABIDE_I_BIDS/derivatives/MNI/sub-51606/anat/sub-51606_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz")
    img_disp = sitk.ReadImage("sim_BN_atrophy.nii.gz")  # This is the transformed brain image
    disp_field = sitk.ReadImage("sim_BN_dispfield.nii.gz")  # This is the displacement field (3D vector)
    jacobian_img = sitk.ReadImage("sim_BN_targetJac.nii.gz")  # This is the target jacobian

    # # Convert SimpleITK images to numpy arrays for plotting
    img_np = sitk.GetArrayFromImage(img)
    img_disp_np = sitk.GetArrayFromImage(img_disp)
    disp_field_np = sitk.GetArrayFromImage(disp_field)

    print(f"Original image shape: {img_np.shape}")
    print(f"Transformed image shape: {img_disp_np.shape}")
    print(f"Displacement field shape: {disp_field_np.shape}")
    print(f"Displacement field data type: {disp_field_np.dtype}")

    # Get middle slice
    mid_slice = img_np.shape[0] // 2

    # difference of the two images
    diff_np = img_disp_np - img_np

    # Get the jacobian image for overlay
    jac_display = sitk.GetArrayFromImage(jacobian_img)

    # Create a figure with subplots
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 5, 1)
    plt.imshow(img_np[mid_slice, :, :], cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(img_disp_np[mid_slice, :, :], cmap='gray')
    plt.title("Transformed Image")
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(diff_np[mid_slice, :, :], cmap='RdBu_r')
    plt.title("Difference Image")
    plt.axis('off')
    plt.colorbar()

    plt.subplot(1, 5, 4)
    # Display displacement field magnitude
    if len(disp_field_np.shape) == 4:
        # 4D displacement field: (z, y, x, components)
        disp_magnitude = np.sqrt(np.sum(disp_field_np[mid_slice, :, :, :]**2, axis=-1))
    else:
        # Handle other cases
        disp_magnitude = np.abs(disp_field_np[mid_slice, :, :])

    plt.imshow(disp_magnitude, cmap='hot')
    plt.title("Displacement Magnitude")
    plt.axis('off')
    plt.colorbar()

    plt.subplot(1, 5, 5)
    # Draw jacobian map over original image
    plt.imshow(img_np[mid_slice, :, :], cmap='gray')
    # Create a mask for non-unity jacobian values
    jac_slice = jac_display[mid_slice, :, :]
    jac_mask = (jac_slice != 1.0)
    # Overlay jacobian map only where it's different from 1
    plt.imshow(np.ma.masked_where(~jac_mask, jac_slice),
              cmap='RdBu_r', alpha=0.6, vmin=0.8, vmax=1.2)
    plt.title("Jacobian Overlay")
    plt.axis('off')
    plt.colorbar(label='Jacobian Value')

    plt.tight_layout()
    plt.savefig("brain_analysis_visualization.png", dpi=150, bbox_inches='tight')
    plt.show()

    # print("Visualization completed!")
