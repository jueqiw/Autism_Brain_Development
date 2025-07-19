import time
import glob
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import monai
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import CenterCrop
from monai.networks.blocks.patchembedding import PatchEmbed, PatchEmbeddingBlock

from utils.const import TENSORBOARD_LOG_DIR, ACE_PHENOTYPE
from utils.utils import (
    seed_everything,
    get_ABIDE_I_subject,
    get_ABIDE_II_subject,
    add_log,
    get_ACE_subjects,
)
from utils.add_argument import add_argument
from utils.transforms import get_train_data_transform
from models.sMRI_encoder import IMG_ENCODER
from models.autoencoder import AutoEncoder

torch.multiprocessing.set_sharing_strategy("file_system")


def main(hparams: Namespace, writer: SummaryWriter):
    train_samples, val_samples = get_ABIDE_I_subject(train_percent=0.9)
    samples = get_ABIDE_II_subject()

    train_transform = get_train_data_transform()
    # val_transform = get_val_data_augmentation_transform()
    train_ds = Dataset(data=samples, transform=train_transform)
    train_loader = DataLoader(
        train_ds, batch_size=hparams.batch_size, shuffle=True, num_workers=8
    )
    # val_ds = Dataset(data=val_samples, transform=train_transform)
    # val_loader = DataLoader(val_ds, batch_size=36, shuffle=False, num_workers=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ACE_samples = get_ACE_subjects()
    test_ds = Dataset(data=ACE_samples, transform=train_transform)
    test_loader = DataLoader(
        test_ds, batch_size=hparams.batch_size, shuffle=False, num_workers=8
    )
    patch_embed_block = PatchEmbeddingBlock(
        in_channels=1,
        img_size=(196, 196, 196),  # assuming the input image size is 196x196x196
        patch_size=(28, 28, 28),
        hidden_size=192,
        num_heads=4,
        proj_type="conv",
        pos_embed_type="sincos",
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

            patch_embed_block.to(device)
            out_patch = patch_embed_block(input_imgs)
            print(f"out_patch shape: {out_patch.shape}")
            sys.exit(0)

            # input_imgs = np.squeeze(input_imgs, axis=1)
            # input_imgs = np.transpose(input_imgs, (0, 2, 3, 1))

            # for i in range(input_imgs.shape[0]):
            #     img = input_imgs[i]
            #     # draw the three dimensional image

            #     cx, cy, cz = np.array(img.shape) // 2
            #     sagittal = np.rot90(img[cx, :, :])  # sagittal slice: left–right axis
            #     coronal = np.rot90(img[:, cy, :])  # coronal  slice: front–back axis
            #     axial = np.rot90(img[:, :, cz])  # axial    slice: top–bottom axis

            #     break

            # test_input_imgs = test_input_imgs * test_mask_imgs
            # test_input_imgs = test_input_imgs.cpu().numpy()

            # test_input_imgs = np.squeeze(test_input_imgs, axis=1)
            # test_input_imgs = np.transpose(test_input_imgs, (0, 2, 3, 1))

            # for i in range(test_input_imgs.shape[0]):
            #     test_img = test_input_imgs[i]
            #     # draw the three dimensional image

            #     cx, cy, cz = np.array(test_img.shape) // 2
            #     sagittal_ace = np.rot90(test_img[cx, :, :])
            #     coronal_ace = np.rot90(test_img[:, cy, :])
            #     axial_ace = np.rot90(test_img[:, :, cz])

            #     break

            # fig, axes = plt.subplots(2, 3, figsize=(9, 6), dpi=300)

            # # draw six images in a 2x3 grid
            # # first row: sagittal, coronal, axial of ABIDE_I
            # for ax, slc in zip(axes[0], [sagittal, coronal, axial]):
            #     ax.imshow(
            #         slc,
            #         cmap="gray",
            #         vmin=np.percentile(img, 1),
            #         vmax=np.percentile(img, 99),
            #     )
            #     ax.axis("off")

            # for ax, slc in zip(axes[1], [sagittal_ace, coronal_ace, axial_ace]):
            #     ax.imshow(
            #         slc,
            #         cmap="gray",
            #         vmin=np.percentile(img, 1),
            #         vmax=np.percentile(img, 99),
            #     )
            #     ax.axis("off")

            # # squeeze the panels together
            # plt.subplots_adjust(wspace=0, hspace=0)  # no gaps
            # plt.tight_layout(pad=0)  # no outer margin

            # # ---------- 4. save ----------
            # out_file = f"/projectnb/ace-ig/jueqiw/experiment/CrossModalityLearning/ABIDE/img_ABIDE_{train_batch_data['subject_id']}_ACE_{test_batch_data['subject_id']}.png"
            # plt.savefig(out_file, bbox_inches="tight", pad_inches=0)
            # plt.show()
            # plt.close()

        break

    #     y = batch_data["label"].to(device)
    #     # create two class one hot label from y
    #     y_onehot = F.one_hot(y, num_classes=2)  # (N, 2), dtype = int64
    #     y_onehot = y_onehot.float()

    #     pred_y = model(input_imgs)
    #     # bce loss
    #     loss = F.binary_cross_entropy_with_logits(pred_y, y_onehot)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     if step % 10 == 0:
    #         print(
    #             f"Epoch {epoch} step {step} loss: {loss.item()}",
    #             flush=True,
    #         )
    #         writer.add_scalar(
    #             "train/loss", loss.item(), epoch * len(train_loader) + step
    #         )

    # model.eval()
    # with torch.no_grad():
    #     y_pred_list = []
    #     y_true_list = []
    #     for step, batch_data in enumerate(val_loader):
    #         input_imgs = batch_data["img"].to(device)
    #         y = batch_data["label"].to(device)
    #         y_onehot = F.one_hot(y, num_classes=2)  # (N, 2), dtype = int64
    #         y_onehot = y_onehot.float()

    #         pred_y = model(input_imgs)
    #         loss = F.binary_cross_entropy_with_logits(pred_y, y_onehot)
    #         y_pred_idx = torch.argmax(pred_y, dim=1)
    #         y_pred_list.append(y_pred_idx)
    #         y_true_list.append(y.cpu().numpy())

    #         if step % 10 == 0:
    #             print(
    #                 f"Epoch {epoch} step {step} loss: {loss.item()}",
    #                 flush=True,
    #             )
    #             writer.add_scalar(
    #                 "val/loss", loss.item(), epoch * len(val_loader) + step
    #             )
    #             writer.add_scalar(
    #                 "val/accuracy",
    #                 (y_pred_idx == y).float().mean().item(),
    #                 epoch * len(val_loader) + step,
    #             )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(42)
    parser = ArgumentParser(description="Trainer args", add_help=False)
    add_argument(parser)
    hparams = parser.parse_args()
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR / hparams.experiment_name)
    main(hparams, writer)
