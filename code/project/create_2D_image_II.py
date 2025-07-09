import sys
import re
from pathlib import Path

import monai
import numpy as np
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import matplotlib.pyplot as plt
import nibabel as nib


from utils.const import ABIDE_I_MNI, ABIDE_II_MNI
from utils.transforms import get_train_data_transform


def create_2D_image(path: Path) -> list:
    imgs = ABIDE_II_MNI.glob(
        "**/sub-*_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz"
    )

    sample_dicts = []
    for img in imgs:
        mask_path = str(img).replace("preproc_T1w", "brain_mask")
        m = re.search(r"sub-(\d{5})(?=_)", img.name)
        if m:
            sample_dicts.append(
                {"sample": m.group(1), "img": str(img), "mask": mask_path}
            )

    return sample_dicts


if __name__ == "__main__":
    dataset_name = "II"
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
                # axial = np.flipud(axial)
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
