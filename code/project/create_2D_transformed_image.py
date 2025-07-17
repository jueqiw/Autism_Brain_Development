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
from utils.transforms import get_train_data_transform


def create_2D_image(path: Path) -> list:
    imgs = path.glob("**/transformed_*.nii.gz")

    ABIDE_I_phenotype_file = pd.read_csv(ABIDE_I_PHENOTYPE)
    sample_dicts = []
    for path in imgs:
        m = re.search(r"-(\d{5})(?=/)", str(path))
        if m:
            subject_id = m.group(1)

        print(f"Processing subject: {subject_id}")

        sys.exit(0)  # For debugging purposes, remove this line in production

        # get the row of ABIDE_I_phenotype_file with the subject_id == SUB_ID
        row = ABIDE_I_phenotype_file[
            ABIDE_I_phenotype_file["SUB_ID"] == int(subject_id)
        ]

        # replace part of the string
        mask_path = str(path).replace("preproc_T1w", "brain_mask")

        # for path in ABIDE_II:
        #     m = re.search(r"_(\d{5})(?=[/_])", str(path))
        #     if m:
        #         subject_id = m.group(1)

        #     pos_base = str(path).find("baseline")
        #     pos_foll = str(path).find("followup")
        #     if pos_base != -1:
        #         row = ABIDE_II_long_phenotype_file[
        #             ABIDE_II_long_phenotype_file["SUB_ID"] == int(subject_id)
        #         ]
        #     elif pos_foll != -1:
        #         continue
        #     else:
        #         row = ABIDE_II_phenotype_file[
        #             ABIDE_II_phenotype_file["SUB_ID"] == int(subject_id)
        #         ]

        sample_dicts.append(
            {
                "img": str(path),
                "mask": mask_path,
                "label": int(row["DX_GROUP"].values[0]) - 1,
                "subject_id": subject_id,
                "site_id": row["SITE_ID"].values[0],
            }
        )

    n_train_samples = int(len(sample_dicts) * train_percent)
    train_samples, val_samples = (
        sample_dicts[:n_train_samples],
        sample_dicts[n_train_samples:],
    )

    return train_samples, val_samples


if __name__ == "__main__":
    dataset_name = "I"
    if dataset_name == "I":
        sample_dicts = create_2D_image(ABIDE_I_MNI)
    elif dataset_name == "II":
        sample_dicts = create_2D_image(ABIDE_II_MNI)

    train_transform = get_train_data_transform()
    train_ds = Dataset(data=sample_dicts, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8)

    n_epochs = 1

    # make folder structure
    ABIDE_2D = Path(f"/projectnb/ace-ig/ABIDE/ABIDE_{dataset_name}_2D")
    ABIDE_2D.mkdir(parents=True, exist_ok=True)
    sagittal_dir = Path(f"/projectnb/ace-ig/ABIDE/ABIDE_{dataset_name}_2D/sagittal")
    sagittal_dir.mkdir(parents=True, exist_ok=True)
    coronal_dir = Path(f"/projectnb/ace-ig/ABIDE/ABIDE_{dataset_name}_2D/coronal")
    coronal_dir.mkdir(parents=True, exist_ok=True)
    axial_dir = Path(f"/projectnb/ace-ig/ABIDE/ABIDE_{dataset_name}_2D/axial")
    axial_dir.mkdir(parents=True, exist_ok=True)

    for step, train_batch_data in enumerate(train_loader):
        input_imgs = train_batch_data["img"]
        mask_imgs = train_batch_data["mask"]

        # apply the mask_imgs to input_imgs
        input_imgs = input_imgs * mask_imgs
        input_imgs = input_imgs.cpu().numpy()

        input_imgs = np.squeeze(input_imgs, axis=1)
        input_imgs = np.transpose(input_imgs, (0, 2, 3, 1))

        for i in range(input_imgs.shape[0]):
            img = input_imgs[i]
            cx, cy, cz = np.array(img.shape) // 2

            for j in range(-2, 3):
                sagittal = np.rot90(img[cx + j, :, :], k=2)
                # rotate 180 degrees to get sagittal view
                coronal = np.rot90(img[:, cy + j, :], k=2)
                axial = np.rot90(img[:, :, cz + j], k=1)
                # flip the axial view to get the correct orientation
                axial = np.fliplr(axial)

                # save as separate images
                plt.imsave(
                    f"/projectnb/ace-ig/ABIDE/ABIDE_{dataset_name}_2D/coronal/{train_batch_data['sample']}_{j}.png",
                    sagittal,
                    cmap="gray",
                )
                plt.imsave(
                    f"/projectnb/ace-ig/ABIDE/ABIDE_{dataset_name}_2D/axial/{train_batch_data['sample']}_{j}.png",
                    coronal,
                    cmap="gray",
                )
                plt.imsave(
                    f"/projectnb/ace-ig/ABIDE/ABIDE_{dataset_name}_2D/sagittal/{train_batch_data['sample']}_{j}.png",
                    axial,
                    cmap="gray",
                )
