import sys
import re
from pathlib import Path
import sys

import monai
import numpy as np
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd

from utils.const import (
    ABIDE_I_MNI,
    ABIDE_II_MNI,
    ABIDE_I_transform,
    ABIDE_II_transform,
    ABIDE_I_PHENOTYPE,
    ABIDE_II_PHENOTYPE,
)
from utils.transforms import get_transformed_train_data_transform
from utils.utils import get_ABIDE_I_transformed_subject


def save_images(img, filename_prefix, train_batch_data, digits):
    cx, cy, cz = img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2
    for j in range(-2, 3):
        sagittal = np.rot90(img[cx + j, :, :], k=2)
        coronal = np.rot90(img[:, cy + j, :], k=2)
        axial = np.rot90(img[:, :, cz + j], k=1)
        axial = np.fliplr(axial)

        # save to npz file
        np.savez_compressed(
            f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/sagittal/{filename_prefix}/{train_batch_data['subject_id'][0]}_{filename_prefix}_{digits}_{j}.npz",
            sagittal=sagittal,
        )
        np.savez_compressed(
            f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/coronal/{filename_prefix}/{train_batch_data['subject_id'][0]}_{filename_prefix}_{digits}_{j}.npz",
            coronal=coronal,
        )
        np.savez_compressed(
            f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/axial/{filename_prefix}/{train_batch_data['subject_id'][0]}_{filename_prefix}_{digits}_{j}.npz",
            axial=axial,
        )


if __name__ == "__main__":
    dataset_name = "I"
    sample_dicts = get_ABIDE_I_transformed_subject()

    train_transform = get_transformed_train_data_transform()
    train_ds = Dataset(data=sample_dicts, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8)
    n_epochs = 1

    # make folder structure
    ABIDE_2D = Path(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained"
    )
    ABIDE_2D.mkdir(parents=True, exist_ok=True)
    sagittal_dir = Path(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/sagittal"
    )
    sagittal_dir.mkdir(parents=True, exist_ok=True)
    coronal_dir = Path(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/coronal"
    )
    coronal_dir.mkdir(parents=True, exist_ok=True)
    axial_dir = Path(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/axial"
    )
    axial_dir.mkdir(parents=True, exist_ok=True)
    sagittal_original_dir = Path(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/sagittal/original"
    )
    sagittal_original_dir.mkdir(parents=True, exist_ok=True)
    coronal_original_dir = Path(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/coronal/original"
    )
    coronal_original_dir.mkdir(parents=True, exist_ok=True)
    axial_original_dir = Path(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/axial/original"
    )
    axial_original_dir.mkdir(parents=True, exist_ok=True)
    # transformed images folder
    sagittal_transformed_dir = Path(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/sagittal/transformed"
    )
    sagittal_transformed_dir.mkdir(parents=True, exist_ok=True)
    coronal_transformed_dir = Path(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/coronal/transformed"
    )
    coronal_transformed_dir.mkdir(parents=True, exist_ok=True)
    axial_transformed_dir = Path(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/axial/transformed"
    )
    axial_transformed_dir.mkdir(parents=True, exist_ok=True)
    # Jacobian images folder
    sagittal_jacobian_dir = Path(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/sagittal/jacobian"
    )
    sagittal_jacobian_dir.mkdir(parents=True, exist_ok=True)
    coronal_jacobian_dir = Path(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/coronal/jacobian"
    )
    coronal_jacobian_dir.mkdir(parents=True, exist_ok=True)
    axial_jacobian_dir = Path(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/axial/jacobian"
    )
    axial_jacobian_dir.mkdir(parents=True, exist_ok=True)
    # Displacement field images folder
    sagittal_dispfield_dir = Path(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/sagittal/dispfield"
    )
    sagittal_dispfield_dir.mkdir(parents=True, exist_ok=True)
    coronal_dispfield_dir = Path(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/coronal/dispfield"
    )
    coronal_dispfield_dir.mkdir(parents=True, exist_ok=True)
    axial_dispfield_dir = Path(
        f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/axial/dispfield"
    )
    axial_dispfield_dir.mkdir(parents=True, exist_ok=True)
    dispfield_img_list = []

    for step, train_batch_data in enumerate(train_loader):
        input_imgs = train_batch_data["img"].detach().cpu().numpy()
        mask_imgs = train_batch_data["mask"].detach().cpu().numpy()
        transformed_imgs = train_batch_data["transformed_img"].detach().cpu().numpy()
        jacobian_imgs = train_batch_data["jacobian"].detach().cpu().numpy()
        dispfield_imgs = train_batch_data["dispfield"].detach().cpu().numpy()
        digits = train_batch_data["digits"][0]

        input_imgs = np.squeeze(input_imgs, axis=1)
        transformed_imgs = np.squeeze(transformed_imgs, axis=1)
        input_imgs = np.transpose(input_imgs, (0, 2, 3, 1))
        transformed_imgs = np.transpose(transformed_imgs, (0, 2, 3, 1))
        jacobian_imgs = np.squeeze(jacobian_imgs, axis=1)
        jacobian_imgs = np.transpose(jacobian_imgs, (0, 2, 3, 1))
        dispfield_imgs = np.transpose(dispfield_imgs, (0, 1, 3, 4, 2))

        for i in range(input_imgs.shape[0]):
            img = input_imgs[i]
            transformed_img = transformed_imgs[i]
            jacobian_img = jacobian_imgs[i]
            dispfield_img = dispfield_imgs[i]
            cx, cy, cz = np.array(img.shape) // 2

            save_images(img, "original", train_batch_data, digits)
            save_images(transformed_img, "transformed", train_batch_data, digits)
            save_images(jacobian_img, "jacobian", train_batch_data, digits)

            # save the displacement field images
            cx, cy, cz = (
                dispfield_img.shape[0] // 2,
                dispfield_img.shape[1] // 2,
                dispfield_img.shape[2] // 2,
            )
            for j in range(-2, 3):
                # Extract displacement slices and compute magnitude (keep 2D structure)
                sagittal_slice = dispfield_img[:, cx + j, :, :]
                coronal_slice = dispfield_img[:, :, cy + j, :]
                axial_slice = dispfield_img[:, :, :, cz + j]

                # Compute magnitude: sqrt(dx² + dy² + dz²) for each pixel
                sagittal = np.rot90(sagittal_slice, k=2, axes=(1, 2))
                coronal = np.rot90(coronal_slice, k=2, axes=(1, 2))
                axial = np.rot90(axial_slice, k=1, axes=(1, 2))
                axial = np.flip(axial, axis=2)

                # save to npz file
                np.savez_compressed(
                    f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/sagittal/dispfield/{train_batch_data['subject_id'][0]}_"
                    f"dispfield_{digits}_{j}.npz",
                    sagittal=sagittal,
                )
                np.savez_compressed(
                    f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/coronal/dispfield/{train_batch_data['subject_id'][0]}_"
                    f"dispfield_{digits}_{j}.npz",
                    coronal=coronal,
                )
                np.savez_compressed(
                    f"/projectnb/ace-genetics/ABIDE/ABIDE_{dataset_name}_2D_Syn_pretrained/axial/dispfield/{train_batch_data['subject_id'][0]}_"
                    f"dispfield_{digits}_{j}.npz",
                    axial=axial,
                )
