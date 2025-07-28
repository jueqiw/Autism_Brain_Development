import logging
import os
import shutil
import sys
import tempfile
import random
from argparse import ArgumentParser
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import AutoEncoder, UNet
from monai.networks.layers import Conv, Norm

from fileinput import filename
from PIL import Image
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import cv2

from utils.add_argument import add_argument

torch.set_num_threads(8)
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_jacobian(jacobian):
    # [-1, 1] range
    jacobian_norm = (jacobian - 1) / 0.15
    return jacobian_norm


def postprocess_jacobian(jacobian_norm):
    return jacobian_norm * 0.15 + 1.0


class MRIDataset(CacheDataset):
    def __init__(self, original_imgs, transformed_imgs, jacobian_maps):
        self.original = original_imgs
        self.transformed = transformed_imgs
        self.jacobians = jacobian_maps

    def __len__(self):
        return len(self.original)

    def __getitem__(self, idx):
        orig = self.original[idx][0]  # shape (192, 192)
        trans = self.transformed[idx][0]

        # Stack: original, transformed
        orig_tensor = torch.from_numpy(orig).unsqueeze(0).float()
        trans_tensor = torch.from_numpy(trans).unsqueeze(0).float()
        input_pair = torch.cat([orig_tensor, trans_tensor], dim=0)

        jacobian = self.jacobians[idx]  # shape (1, 192, 192) - already has channel dim
        jacobian = preprocess_jacobian(jacobian)
        jacobian = torch.from_numpy(jacobian).float()
        return input_pair, jacobian


def log_prediction_images(writer, inputs, predictions, targets, epoch, prefix=""):
    """
    Log input images, predictions, and targets to TensorBoard
    inputs: shape (batch, 4, H, W) - 4 channels: orig, trans, sobel(orig), sobel(trans)
    predictions: shape (batch, 1, H, W) - predicted jacobian
    targets: shape (batch, 1, H, W) - ground truth jacobian
    prefix: string to add to the title (e.g., "Train" or "Val")
    """
    # Create a grid showing input channels, prediction, and target for first sample
    sample_idx = 0

    # Extract input channels (original and transformed images)
    orig_img = inputs[sample_idx, 0:1]  # First channel (original)
    trans_img = inputs[sample_idx, 1:2]

    # Get prediction and target
    pred_img = predictions[sample_idx : sample_idx + 1]
    target_img = targets[sample_idx : sample_idx + 1]

    pred_img_post = postprocess_jacobian(pred_img)
    target_img_post = postprocess_jacobian(target_img)

    # Create matplotlib figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(orig_img.squeeze().cpu().numpy(), cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(trans_img.squeeze().cpu().numpy(), cmap="gray")
    axes[1].set_title("Transformed Image")
    axes[1].axis("off")

    axes[2].imshow(pred_img_post.squeeze().cpu().numpy(), cmap="gray")
    axes[2].set_title("Predicted Jacobian")
    axes[2].axis("off")

    axes[3].imshow(target_img_post.squeeze().cpu().numpy(), cmap="gray")
    axes[3].set_title("Target Jacobian")
    axes[3].axis("off")

    plt.tight_layout()

    # Save matplotlib figure to TensorBoard with prefix
    title = (
        f"{prefix}_Comparison_epoch_{epoch}" if prefix else f"Comparison_epoch_{epoch}"
    )
    writer.add_figure(title, fig, epoch)
    plt.close(fig)  # Important: close figure to free memory


def brain_masked_loss(
    pred, target, input_img, brain_threshold=0.02, foreground_weight=10.0
):
    """
    Compute loss only within brain regions (input > brain_threshold)
    Give higher weight to non-zero target regions
    """
    # Create brain mask from input image (average across channels if needed)
    if input_img.dim() == 4:  # (batch, channels, H, W)
        brain_mask = (input_img.mean(dim=1, keepdim=True) > brain_threshold).float()
    else:  # (batch, H, W)
        brain_mask = (input_img > brain_threshold).float()
    # L1 loss
    l1_loss = torch.abs(pred - target)

    # Weight map: higher weight for non-zero targets within brain
    target_weights = torch.where(torch.abs(target) > 0, foreground_weight, 1.0)

    # Combine brain mask and target weights
    final_weights = brain_mask * target_weights

    # Apply weights and compute masked loss
    masked_loss = final_weights * l1_loss

    # Only compute mean over brain regions (avoid division by zero)
    brain_pixels = brain_mask.sum()
    if brain_pixels > 0:
        return masked_loss.sum() / brain_pixels
    else:
        return masked_loss.mean()  # Fallback


def train_model(
    train_loader,
    val_loader,
    model,
    epochs=200,
    lr=1e-4,
    log_dir="/projectnb/ace-genetics/jueqiw/experiment/Autism_Brain_Development/experiments/tensorboard",
    hparams=None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    tanh = nn.Tanh()

    # Initialize TensorBoard writer
    # create new folder for each run
    if hparams and hasattr(hparams, "experiment_name"):
        log_dir = os.path.join(log_dir, hparams.experiment_name)
    writer = SummaryWriter(log_dir=log_dir)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        total_train_loss = 0
        model.train()

        # Store training samples for logging (much more memory efficient)
        train_sample_for_logging = None

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            output = tanh(output)  # Apply tanh to output

            loss = brain_masked_loss(output, y, x, brain_threshold=0.05)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            # Clear GPU cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            # Only store one batch for logging and immediately move to CPU to save GPU memory
            if batch_idx == 0 and (epoch + 1) % 10 == 0:  # Only collect when needed
                train_sample_for_logging = (
                    x[:4].detach().cpu(),  # Only first 4 samples, move to CPU
                    output[:4].detach().cpu(),
                    y[:4].detach().cpu(),
                )

        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Log training loss to TensorBoard
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        # Validation step
        model.eval()
        total_val_loss = 0
        sample_inputs = sample_predictions = sample_targets = None

        with torch.no_grad():
            for batch_idx, (x_val, y_val) in enumerate(val_loader):
                x_val, y_val = x_val.to(device), y_val.to(device)
                output_val = model(x_val)
                output_val = tanh(output_val)

                # Fix shape mismatch - squeeze extra dimensions
                if output_val.dim() > y_val.dim():
                    output_val = output_val.squeeze()

                val_loss = brain_masked_loss(
                    output_val, y_val, x_val, brain_threshold=0.05
                )
                total_val_loss += val_loss.item()

                # Only store one batch for logging and move to CPU to save memory
                if batch_idx == 4 and (epoch + 1) % 10 == 0:
                    random_indices = torch.randperm(x_val.size(0))
                    sample_inputs = x_val[random_indices].detach().cpu()
                    sample_predictions = output_val[random_indices].detach().cpu()
                    sample_targets = y_val[random_indices].detach().cpu()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        # Log validation loss to TensorBoard
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

        # Log prediction images every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Log training images
            if train_sample_for_logging is not None:
                sample_train_inputs, sample_train_predictions, sample_train_targets = (
                    train_sample_for_logging
                )

                # Move back to GPU temporarily for logging
                sample_train_inputs = sample_train_inputs.to(device)
                sample_train_predictions = sample_train_predictions.to(device)
                sample_train_targets = sample_train_targets.to(device)

                # Log training images
                log_prediction_images(
                    writer,
                    sample_train_inputs,
                    sample_train_predictions,
                    sample_train_targets,
                    epoch,
                    prefix="Train",
                )

            # Log validation images
            if sample_inputs is not None:
                sample_inputs = sample_inputs.to(device)
                sample_predictions = sample_predictions.to(device)
                sample_targets = sample_targets.to(device)

                log_prediction_images(
                    writer,
                    sample_inputs,
                    sample_predictions,
                    sample_targets,
                    epoch,
                    prefix="Val",
                )

    writer.close()

    return model


# Loading all data (same logic as TF version)
def get_ages(sub_id, dataset_num):
    sub_id = int(sub_id)
    if dataset_num == 1:
        df = pd.read_csv("../../ace-ig/ABIDE/Phenotypic_V1_0b.csv")
        long_df = pd.read_csv(
            "../../ace-ig/ABIDE/ABIDEII_Long_Composite_Phenotypic.csv"
        )  # longitudinal subjects
        if not long_df[
            (long_df["SUB_ID"] == sub_id)
        ].empty:  # subject found in ABIDE II longitudinal subjects - remove
            return None
        age = df[(df["SUB_ID"] == sub_id)]["AGE_AT_SCAN"].values[0]
        if age > 21:
            return None
    elif dataset_num == 2:
        df = pd.read_csv(
            "../../ace-ig/ABIDE/ABIDEII_Composite_Phenotypic.csv", encoding="cp1252"
        )
        if df[(df["SUB_ID"] == sub_id)].empty:
            return None
        age = df[(df["SUB_ID"] == sub_id)]["AGE_AT_SCAN "].values[0]  # note extra space
        if age > 21:
            return None
    return age


def load_data():
    folder_paths = [
        # "/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/axial/original",
        "/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/coronal/original",
        # "/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/sagittal/original",
        # "/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/axial/transformed",
        "/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/coronal/transformed",
        # "/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/sagittal/transformed",
        # "/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/axial/jacobian",
        "/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/coronal/jacobian",
        # "/projectnb/ace-genetics/ABIDE/ABIDE_I_2D_Syn_pretrained/sagittal/jacobian",
        # "/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/axial/original",
        "/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/coronal/original",
        # "/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/sagittal/original",
        # "/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/axial/transformed",
        "/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/coronal/transformed",
        # "/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/sagittal/transformed",
        # "/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/axial/jacobian",
        "/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/coronal/jacobian",
        # "/projectnb/ace-genetics/ABIDE/ABIDE_II_2D_Syn_pretrained/sagittal/jacobian",
    ]
    original_list = []
    transformed_list = []
    jacobian_list = []

    for i in range(0, len(folder_paths), 3):
        original_folder = folder_paths[i]
        transformed_folder = folder_paths[i + 1]
        jacobian_folder = folder_paths[i + 2]
        dataset_num = 1 if "ABIDE_I_2D" in original_folder else 2

        print(f"Processing ABIDE {dataset_num} folders.")

        for filename in os.listdir(original_folder):
            if not filename.endswith(".npz"):
                continue

            # Load original, transformed, and jacobian images
            original_path = os.path.join(original_folder, filename)
            transformed_filename = filename.replace("original", "transformed")
            jacobian_filename = filename.replace("original", "jacobian")
            transformed_path = os.path.join(transformed_folder, transformed_filename)
            jacobian_path = os.path.join(jacobian_folder, jacobian_filename)

            if not (os.path.exists(transformed_path) and os.path.exists(jacobian_path)):
                continue  # skip if anything is missing for that subject

            # Load original npz (assumed key: 'coronal')
            original_data = np.load(original_path)
            original_img = original_data["coronal"].astype(np.float32)
            original_img = cv2.resize(
                original_img, (192, 192), interpolation=cv2.INTER_LINEAR
            )

            # Load transformed npz (assumed key: 'coronal')
            transformed_data = np.load(transformed_path)
            transformed_img = transformed_data["coronal"].astype(np.float32)
            transformed_img = cv2.resize(
                transformed_img, (192, 192), interpolation=cv2.INTER_LINEAR
            )

            # Load jacobian npz (assumed key: 'coronal')
            jacobian_data = np.load(jacobian_path)
            jacobian_img = jacobian_data["coronal"].astype(np.float32)
            jacobian_img = cv2.resize(
                jacobian_img, (192, 192), interpolation=cv2.INTER_LINEAR
            )

            # Add channels
            original_list.append(
                original_img[np.newaxis, :, :]
            )  # add batch and channel dims (1, 1, 192, 192)
            transformed_list.append(transformed_img[np.newaxis, :, :])
            jacobian_list.append(jacobian_img[np.newaxis, :, :])

    original_array = np.array(original_list, dtype=np.float32)
    transformed_array = np.array(transformed_list, dtype=np.float32)
    jacobian_array = np.array(jacobian_list, dtype=np.float32)

    print(f"Loaded {original_array.shape[0]} valid samples.")
    return original_array, transformed_array, jacobian_array


class CustomAutoEncoder(nn.Module):
    def __init__(
        self, spatial_dims, in_channels, out_channels, channels, strides, num_res_units
    ):
        super().__init__()

        # First layer with InstanceNorm and PReLU
        self.first_conv = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels,
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            norm=Norm.INSTANCE,
            act="PRELU",
        )

        # Base autoencoder (modify input channels since first layer is handled separately)
        self.autoencoder = AutoEncoder(
            spatial_dims=spatial_dims,
            in_channels=channels[0],
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
        )

    def forward(self, x):
        # Pass through first layer with InstanceNorm and PReLU
        x = self.first_conv(x)
        # Pass through rest of autoencoder
        return self.autoencoder(x)


def main(hparams):
    original, transformed, jacobians = load_data()

    # Parse model architecture parameters
    channels = tuple(map(int, hparams.model_channels.split(",")))
    strides = tuple(map(int, hparams.model_strides.split(",")))

    # --model_channels "32,64,64" \
    # --model_strides "2,2,1" \
    # --num_res_units 3 \
    # --model_dropout 0.1 \
    # --batch_size 8

    model = CustomAutoEncoder(
        spatial_dims=2,
        in_channels=2,
        out_channels=1,
        channels=channels,  # "32,64,64"
        strides=strides,  # "2,2,1"
        num_res_units=hparams.num_res_units,  # 3
    ).to(device)

    # Custom initialization to prevent zero outputs
    def init_weights(m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            # Xavier/Glorot initialization for better gradient flow
            torch.nn.init.xavier_normal_(m.weight, gain=1.0)
            if m.bias is not None:
                # Small random bias to break symmetry
                torch.nn.init.uniform_(m.bias, -0.1, 0.1)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)

    model.apply(init_weights)
    print("Applied Xavier initialization with random bias")

    train_orig, val_orig, train_trans, val_trans, train_jac, val_jac = train_test_split(
        original, transformed, jacobians, test_size=0.2, random_state=42
    )

    train_set = MRIDataset(train_orig, train_trans, train_jac)
    val_set = MRIDataset(val_orig, val_trans, val_jac)

    train_loader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=hparams.batch_size, shuffle=False)

    model = train_model(
        train_loader,
        val_loader,
        model,
        epochs=hparams.n_epochs,
        lr=hparams.learning_rate,
        log_dir="/projectnb/ace-genetics/jueqiw/experiment/Autism_Brain_Development/experiments/tensorboard",
        hparams=hparams,
    )

    # save the model
    torch.save(model.state_dict(), f"all_model_monai_{hparams.experiment_name}.pt")


if __name__ == "__main__":
    parser = ArgumentParser(description="Trainer args", add_help=False)
    add_argument(parser)
    main(parser.parse_args())
