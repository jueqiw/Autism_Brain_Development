# train_model.py
import argparse
import pickle
import os
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import disptools.displacements as dsp



def create_samples_for_training(id):
    # load pickle file with samples
    with open("/project/ace-genetics/jueqiw/Autism_Brain_Development/code/project/subjects_ABIDE_lower_35.pkl", "rb") as f:
        samples = pickle.load(f)

    n_jobs = 20
    id_ = int(id)  # convert to zero-based index
    n_subjects = len(samples)
    subjects_per_job = np.ceil(n_subjects / n_jobs).astype(int)
    start_idx = id_ * subjects_per_job
    end_idx = (id_ + 1) * subjects_per_job if id_ < (n_jobs - 1) else n_subjects
    job_subjects = samples[start_idx:end_idx]

    for subject in job_subjects:
        dataset = subject['dataset']
        subject_id = subject['subject_id']

        dir_path = Path(f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset}/sub-{subject_id}")
        dir_path.mkdir(parents=True, exist_ok=True)
        # Use the actual Brainnetome atlas with parcel labels
        atlas = sitk.ReadImage("/projectnb/ace-genetics/jueqiw/dataset/MRI_template/Atlas/BN_Atlas_246_space-MNI152NLin2009cAsym.nii.gz")
        
        # Debug: Check what labels exist in the atlas
        lab_np = sitk.GetArrayFromImage(atlas)
        unique_labels = np.unique(lab_np)

        for idx in range(3):
            print(f"Processing subject {subject_id}, job {id_}, index {idx}")
            # compute the running time
            start_time = time.time()
            random_seed = int(time.time() * 1000000) % (2**32)  # Use current time as seed
            rng = np.random.default_rng(seed=random_seed)
            all_labels = np.arange(1, 247)
            n_rois = 20
            chosen = rng.choice(all_labels, n_rois, replace=False)
            atrophy_labels = chosen[: n_rois // 2]
            expand_labels = chosen[n_rois // 2 :]


            # More conservative deformation factors to avoid instability
            atrophy_factor = rng.uniform(0.85, 1, size=n_rois // 2)  # Less extreme atrophy
            expand_factor = rng.uniform(1, 1.15, size=n_rois // 2)   # Less extreme expansion
            lab_np = sitk.GetArrayFromImage(atlas)
            jac_np = np.ones_like(lab_np, dtype=np.float32)

            for roi, factor in zip(atrophy_labels, atrophy_factor):
                jac_np[lab_np == roi] = factor
            for roi, factor in zip(expand_labels, expand_factor):
                jac_np[lab_np == roi] = factor

            jacobian_img = sitk.GetImageFromArray(jac_np)
            jacobian_img.CopyInformation(atlas)
            mask_np = (jac_np != 1.0).astype(np.uint8)
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
            # Cast to Float32 for deep learning efficiency
            disp_cast = sitk.Cast(disp, sitk.sitkVectorFloat32)
            disp_path = dir_path / f"sim_BN_dispfield_{idx}.nii.gz"
            sitk.WriteImage(disp_cast, str(disp_path))
            jacobian_path = dir_path / f"sim_BN_targetJac_{idx}.nii.gz"
            sitk.WriteImage(jacobian_img, str(jacobian_path))

            img_path = subject['img']
            img = sitk.ReadImage(img_path)
            img = sitk.Resample(
                img, atlas, sitk.Transform(), sitk.sitkLinear, 0.0, img.GetPixelIDValue()
            )
            disp_cast = sitk.Cast(disp, sitk.sitkVectorFloat64)
            disp_transform = sitk.DisplacementFieldTransform(disp_cast)
            img_transformed = sitk.Resample(img, disp_transform, sitk.sitkLinear, 0.0, img.GetPixelIDValue())
            img_transformed_path = dir_path / f"transformed_{idx}.nii.gz"
            sitk.WriteImage(img_transformed, str(img_transformed_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True,
                        help="ID or file name of the subject to process")
    args = parser.parse_args()

    id_ = args.id
    create_samples_for_training(id_)


if __name__ == "__main__":
    main()