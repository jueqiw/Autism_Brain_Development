import sys
import re
import monai
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from pathlib import Path
import matplotlib.pyplot as plt
import nibabel as nib

from utils.const import ABIDE_I_MNI, ABIDE_II_MNI
from utils.transforms import get_train_data_transform


def create_2D_image(path: str):
    imgs = Path(path).glob("**/sub-*_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz")

    sample_dicts = []
    for img in imgs:
        m = re.search(r"sub-(\w+)", img.name)
        if m:
            sample_dicts.append({"sample": m.group(1), "img": img})

    return sample_dicts


if __name__ == "__main__":
    sample_dicts = create_2D_image(ABIDE_I_MNI)
    sample_dicts += create_2D_image(ABIDE_II_MNI)

    train_transform = get_train_data_transform()
    train_ds = Dataset(data=sample_dicts, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8)
    # val_ds = Dataset(data=val_samples, transform=train_transform)
    # val_loader = DataLoader(val_ds, batch_size=36, shuffle=False, num_workers=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ACE_samples = get_ACE_subjects()
    test_ds = Dataset(data=ACE_samples, transform=train_transform)
    test_loader = DataLoader(
        test_ds, batch_size=hparams.batch_size, shuffle=False, num_workers=8
    )
    model = AutoEncoder(
        n_embed=8,
        embed_dim=64,
        n_alpha_channels=1,
        n_channels=64,
        n_res_channels=64,
        n_res_layers=2,
        p_dropout=0.1,
        latent_resolution="low",
    )
    model.to(device)

    n_epochs = 1
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hparams.learning_rate, weight_decay=1e-4
    )

    for epoch in range(n_epochs):
        model.train()
        print(f"Epoch {epoch} started")

        for step, (train_batch_data, test_batch_data) in enumerate(
            zip(train_loader, test_loader)
        ):
            input_imgs = train_batch_data["img"].to(device)
            mask_imgs = train_batch_data["mask"].to(device)

            test_input_imgs = test_batch_data["img"].to(device)
            test_mask_imgs = test_batch_data["mask"].to(device)

            # apply the mask_imgs to input_imgs
            input_imgs = input_imgs * mask_imgs
            input_imgs = input_imgs.cpu().numpy()

            input_imgs = np.squeeze(input_imgs, axis=1)
            input_imgs = np.transpose(input_imgs, (0, 2, 3, 1))

            for i in range(input_imgs.shape[0]):
                img = input_imgs[i]
                # draw the three dimensional image

                cx, cy, cz = np.array(img.shape) // 2
                sagittal = np.rot90(img[cx, :, :])  # sagittal slice: left–right axis
                coronal = np.rot90(img[:, cy, :])  # coronal  slice: front–back axis
                axial = np.rot90(img[:, :, cz])  # axial    slice: top–bottom axis

                break
