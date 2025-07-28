import os
import glob
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage
from PIL import Image
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    ScaleIntensityRanged,
    ToTensord,
)
from monai.data import CacheDataset, partition_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.add_argument import add_argument  # Assuming this is the correct import path
from utils.const import (
    ABIDE_I_2D_REGRESSION,
    ABIDE_II_2D_REGRESSION,
    ABIDE_PATH,
    PRE_TRAINED_WEIGHTS,
)
from models.vae import create_vae_model

torch.set_num_threads(8)  # or the number you request
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_transforms():
    """Create MONAI transforms for data preprocessing with intensity normalization"""
    return Compose(
        [
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(192, 192)),
            ScaleIntensityRanged(
                keys=["image"], a_min=0.5, a_max=99.5, b_min=0.0, b_max=1.0, clip=True
            ),
            ToTensord(keys=["image"]),
        ]
    )


# Normalize a single slice (kept for backward compatibility)
def normalize_slice(slice_data):
    lower, upper = np.percentile(slice_data, [0.5, 99.5])
    slice_clipped = np.clip(slice_data, lower, upper)
    normalized = (slice_clipped - lower) / (upper - lower + 1e-8)
    return normalized


class MRIAgeDataset(Dataset):
    """Custom dataset class that wraps MONAI CacheDataset for training"""

    def __init__(self, cached_dataset, indices=None):
        self.cached_dataset = cached_dataset
        self.indices = (
            indices if indices is not None else list(range(len(cached_dataset)))
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the actual index from the subset
        actual_idx = self.indices[idx]
        sample = self.cached_dataset[actual_idx]

        img = sample["image"]  # Already preprocessed tensor
        age = torch.tensor([sample["age"]], dtype=torch.float32)
        return img, age


# Reparameterization trick
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# Encoder module
class Encoder(nn.Module):
    def __init__(self, latent_dim=16, ft_bank_baseline=16, dropout_alpha=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, ft_bank_baseline, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(ft_bank_baseline, ft_bank_baseline * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(ft_bank_baseline * 2, ft_bank_baseline * 4, 3, padding=1)
        self.flatten_size = (192 // 8) * (192 // 8) * ft_bank_baseline * 4

        self.dropout = nn.Dropout(dropout_alpha)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flatten_size, latent_dim * 4)
        self.fc_z_mean = nn.Linear(latent_dim * 4, latent_dim)
        self.fc_z_log_var = nn.Linear(latent_dim * 4, latent_dim)

        self.fc_r_mean = nn.Linear(latent_dim * 4, 1)
        self.fc_r_log_var = nn.Linear(latent_dim * 4, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = torch.tanh(self.fc1(x))

        z_mean = self.fc_z_mean(x)
        z_log_var = self.fc_z_log_var(x)
        r_mean = self.fc_r_mean(x)
        r_log_var = self.fc_r_log_var(x)

        z = reparameterize(z_mean, z_log_var)
        r = reparameterize(r_mean, r_log_var)
        return z_mean, z_log_var, z, r_mean, r_log_var, r


# Generator module (generates latent distribution parameters from age)
class Generator(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.pz_mean = nn.Linear(1, latent_dim)
        self.pz_log_var = nn.Linear(1, 1)

    def forward(self, r):
        pz_mean = self.pz_mean(r)
        pz_log_var = self.pz_log_var(r)
        return pz_mean, pz_log_var


# Decoder module (decodes latent z to image)
class Decoder(nn.Module):
    def __init__(self, latent_dim=16, ft_bank_baseline=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.ft_bank_baseline = ft_bank_baseline

        downsampled_H = 192 // 8
        downsampled_W = 192 // 8
        self.flattened_size = downsampled_H * downsampled_W * ft_bank_baseline * 4

        self.fc1 = nn.Linear(latent_dim, latent_dim * 2)
        self.fc2 = nn.Linear(latent_dim * 2, latent_dim * 4)
        self.fc3 = nn.Linear(latent_dim * 4, self.flattened_size)

        self.conv1 = nn.Conv2d(ft_bank_baseline * 4, ft_bank_baseline * 4, 3, padding=1)
        self.conv2 = nn.Conv2d(ft_bank_baseline * 4, ft_bank_baseline * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(ft_bank_baseline * 2, ft_bank_baseline, 3, padding=1)
        self.conv4 = nn.Conv2d(ft_bank_baseline, 1, 3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, z):
        x = torch.tanh(self.fc1(z))
        x = torch.tanh(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, self.ft_bank_baseline * 4, 192 // 8, 192 // 8)

        x = F.relu(self.conv1(x))
        x = self.upsample(x)
        x = F.relu(self.conv2(x))
        x = self.upsample(x)
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = self.conv4(x)
        # No activation if not binary image (can add sigmoid if needed)
        return x


# Loss function combining reconstruction, KL, and label (age) loss
def vae_loss_fn(
    x, x_recon, z_mean, z_log_var, pz_mean, pz_log_var, r_mean, r_log_var, r
):
    # Reconstruction loss (MAE)
    recon_loss = F.l1_loss(x_recon, x, reduction="mean")

    # KL divergence between posterior z and prior pz
    kl_loss = (
        1
        + z_log_var
        - pz_log_var
        - ((z_mean - pz_mean).pow(2) / pz_log_var.exp())
        - (z_log_var.exp() / pz_log_var.exp())
    )
    kl_loss = -0.5 * kl_loss.sum(dim=1).mean()

    # Label loss (age prediction)
    label_loss = 0.5 * ((r_mean - r).pow(2) / r_log_var.exp()) + 0.5 * r_log_var
    label_loss = label_loss.mean()

    return recon_loss + kl_loss + label_loss


# Augmentation functions removed - no data augmentation used


# Loading all data (same logic as TF version)
def get_ages(sub_id, dataset_num):
    sub_id = int(sub_id)
    if dataset_num == 1:
        df = pd.read_csv(ABIDE_PATH / "Phenotypic_V1_0b.csv")
        age = df[(df["SUB_ID"] == sub_id)]["AGE_AT_SCAN"].values[0]
        if age > 21:
            return None
    elif dataset_num == 2:
        df = pd.read_csv(
            ABIDE_PATH / "ABIDEII_Composite_Phenotypic.csv", encoding="cp1252"
        )
        if df[(df["SUB_ID"] == sub_id)].empty:  #
            return None
        age = df[(df["SUB_ID"] == sub_id)]["AGE_AT_SCAN "].values[0]  # note extra space
        if age > 21:
            return None
    return age


def prepare_data_dicts():
    """Prepare data dictionaries for MONAI CacheDataset"""
    folder_paths = [
        ABIDE_I_2D_REGRESSION / "axial",
        ABIDE_II_2D_REGRESSION / "axial",
    ]

    data_dicts = []

    for folder_path in folder_paths:
        if not folder_path.exists():
            print(f"Warning: Folder {folder_path} does not exist, skipping...")
            continue

        print(f"Scanning folder: {folder_path}")
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                subject_id = filename[2:7]
                dataset_num = 1 if "ABIDE_I_2D" in str(folder_path) else 2
                age = get_ages(subject_id, dataset_num)

                if age is None:
                    continue

                img_path = str(folder_path / filename)

                data_dicts.append(
                    {
                        "image": img_path,
                        "age": float(age),
                        "subject_id": subject_id,
                        "dataset_num": dataset_num,
                    }
                )

    print(f"Found {len(data_dicts)} valid samples")
    return data_dicts


def create_monai_dataset(cache_rate=1.0, num_workers=4):
    """Create MONAI CacheDataset for data loading"""

    # Prepare data dictionaries
    data_dicts = prepare_data_dicts()

    if len(data_dicts) == 0:
        raise ValueError("No valid data found!")

    # Create single cache dataset (no augmentation)
    print("Creating MONAI CacheDataset...")
    cached_ds = CacheDataset(
        data=data_dicts,
        transform=get_data_transforms(),
        cache_rate=cache_rate,
        num_workers=num_workers,
        progress=True,
    )

    return cached_ds


def save_image_tensor(img_tensor, filename):
    # img_tensor shape: (1, H, W)
    img_np = img_tensor.squeeze().cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    img.save(filename)


def log_vae_images(writer, inputs, reconstructions, ages, epoch, prefix=""):
    """
    Log VAE input and reconstruction images to TensorBoard with age info
    """
    sample_idx = 0

    # Get first sample
    input_img = inputs[sample_idx : sample_idx + 1]  # (1, 1, H, W)
    recon_img = reconstructions[sample_idx : sample_idx + 1]  # (1, 1, H, W)
    age = ages[sample_idx].item()

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Input image
    axes[0].imshow(input_img.squeeze().cpu().numpy(), cmap="gray")
    axes[0].set_title(f"Input (Age: {age:.1f})")
    axes[0].axis("off")

    # Reconstruction
    axes[1].imshow(recon_img.squeeze().cpu().numpy(), cmap="gray")
    axes[1].set_title("VAE Reconstruction")
    axes[1].axis("off")

    plt.tight_layout()

    title = (
        f"{prefix}_VAE_Images_epoch_{epoch}" if prefix else f"VAE_Images_epoch_{epoch}"
    )
    writer.add_figure(title, fig, epoch)
    plt.close(fig)


def train_fold_two_stage(
    train_loader, val_loader, model, epochs, lr=1e-3, hparams=None
):
    """
    Stage 1: Warm-up with frozen encoder (train only VAE heads)
    Stage 2: Gradual fine-tuning with layer-by-layer unfreezing
    """
    model.train()

    # Setup TensorBoard writer
    log_dir = "/projectnb/ace-genetics/jueqiw/experiment/Autism_Brain_Development/experiments/tensorboard"
    if hparams and hasattr(hparams, "experiment_name"):
        log_dir = os.path.join(log_dir, f"{hparams.experiment_name}_two_stage")
    writer = SummaryWriter(log_dir=log_dir)

    warmup_epochs = epochs // 4  # 25% for warm-up
    finetune_epochs = epochs - warmup_epochs
    stage_epochs = finetune_epochs // 3  # Divide fine-tuning into 3 stages

    print(
        f"Two-stage training: {warmup_epochs} warm-up + {finetune_epochs} fine-tuning epochs"
    )

    total_train_history = []
    total_val_history = []

    # ========== STAGE 1: WARM-UP ==========
    print("Stage 1: Warm-up with frozen encoder...")
    model.freeze_pretrained_encoder()
    model.freeze_pretrained_decoder()

    param_groups = model.get_parameter_groups(
        new_lr=lr, pretrained_lr=lr / 10, weight_decay=1e-4
    )
    optimizer = torch.optim.Adam(param_groups)

    for epoch in range(warmup_epochs):
        train_loss, val_loss = train_epoch(
            train_loader, val_loader, model, optimizer, writer, epoch, "warmup"
        )
        total_train_history.append(train_loss)
        total_val_history.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Warm-up Epoch {epoch+1}/{warmup_epochs} - Train: {train_loss:.6f} | Val: {val_loss:.6f}"
            )

    print("Stage 2: Gradual fine-tuning...")

    for stage in range(1, 4):  # 3 fine-tuning stages
        print(f"Fine-tuning stage {stage}/3 - Unfreezing deeper layers...")

        # Gradually unfreeze layers
        model.unfreeze_encoder_layer_by_layer(stage)

        param_groups = model.get_parameter_groups(
            new_lr=lr * 0.5,  # Lower LR for new layers in fine-tuning
            pretrained_lr=lr * 0.1,  # Much lower LR for pretrained layers
            weight_decay=1e-4,  # L2 regularization for pretrained layers
        )
        optimizer = torch.optim.Adam(param_groups)

        # Train for this stage
        stage_start_epoch = warmup_epochs + (stage - 1) * stage_epochs
        stage_end_epoch = min(warmup_epochs + stage * stage_epochs, epochs)

        for epoch in range(stage_start_epoch, stage_end_epoch):
            train_loss, val_loss = train_epoch(
                train_loader,
                val_loader,
                model,
                optimizer,
                writer,
                epoch,
                f"finetune_stage_{stage}",
            )
            total_train_history.append(train_loss)
            total_val_history.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Fine-tune Stage {stage} Epoch {epoch+1-stage_start_epoch+1}/{stage_end_epoch-stage_start_epoch} - Train: {train_loss:.6f} | Val: {val_loss:.6f}"
                )

    writer.close()
    return model


def train_epoch(train_loader, val_loader, model, optimizer, writer, epoch, stage_name):
    """Single epoch training and validation"""
    model.train()

    # Training phase
    total_train_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_age_loss = 0

    train_sample_for_logging = None

    for batch_idx, (imgs, ages) in enumerate(train_loader):
        imgs = imgs.to(device)
        ages = ages.to(device)

        optimizer.zero_grad()

        # Forward pass through VAE
        (
            recon,
            z_mean,
            z_logvar,
            pz_mean,
            pz_logvar,
            age_mean,
            age_logvar,
            age_pred,
        ) = model(imgs, ages)

        # Compute VAE loss using local vae_loss_fn
        total_loss = vae_loss_fn(
            imgs,
            recon,
            z_mean,
            z_logvar,
            pz_mean,
            pz_logvar,
            age_mean,
            age_logvar,
            ages,
        )
        recon_loss = kl_loss = age_loss = total_loss  # For logging purposes

        total_loss.backward()
        optimizer.step()

        total_train_loss += total_loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_age_loss += age_loss.item()

        # Store sample for logging
        if batch_idx == 0 and (epoch + 1) % 10 == 0:
            train_sample_for_logging = (
                imgs[:4].detach().cpu(),
                recon[:4].detach().cpu(),
                ages[:4].detach().cpu(),
            )

    avg_train_loss = total_train_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_kl_loss = total_kl_loss / len(train_loader)
    avg_age_loss = total_age_loss / len(train_loader)

    # Validation phase
    model.eval()
    total_val_loss = 0
    sample_inputs = sample_reconstructions = sample_ages = None

    with torch.no_grad():
        for batch_idx, (imgs_val, ages_val) in enumerate(val_loader):
            imgs_val = imgs_val.to(device)
            ages_val = ages_val.to(device)

            (
                recon_val,
                z_mean_val,
                z_logvar_val,
                pz_mean_val,
                pz_logvar_val,
                age_mean_val,
                age_logvar_val,
                age_pred_val,
            ) = model(imgs_val, ages_val)

            val_loss = vae_loss_fn(
                imgs_val,
                recon_val,
                z_mean_val,
                z_logvar_val,
                pz_mean_val,
                pz_logvar_val,
                age_mean_val,
                age_logvar_val,
                ages_val,
            )
            total_val_loss += val_loss.item()

            # Store samples for logging
            if batch_idx == 0 and (epoch + 1) % 10 == 0:
                sample_inputs = imgs_val[:4].detach().cpu()
                sample_reconstructions = recon_val[:4].detach().cpu()
                sample_ages = ages_val[:4].detach().cpu()

    avg_val_loss = total_val_loss / len(val_loader)

    # Log metrics to TensorBoard
    writer.add_scalar(f"Loss/{stage_name}_Train_Total", avg_train_loss, epoch)
    writer.add_scalar(f"Loss/{stage_name}_Train_Reconstruction", avg_recon_loss, epoch)
    writer.add_scalar(f"Loss/{stage_name}_Train_KL", avg_kl_loss, epoch)
    writer.add_scalar(f"Loss/{stage_name}_Train_Age", avg_age_loss, epoch)
    writer.add_scalar(f"Loss/{stage_name}_Validation", avg_val_loss, epoch)

    # Log images every 10 epochs
    if (epoch + 1) % 10 == 0:
        if train_sample_for_logging is not None:
            sample_train_inputs, sample_train_recons, sample_train_ages = (
                train_sample_for_logging
            )
            log_vae_images(
                writer,
                sample_train_inputs,
                sample_train_recons,
                sample_train_ages,
                epoch,
                prefix=f"{stage_name}_Train",
            )

        if sample_inputs is not None:
            log_vae_images(
                writer,
                sample_inputs,
                sample_reconstructions,
                sample_ages,
                epoch,
                prefix=f"{stage_name}_Val",
            )

    return avg_train_loss, avg_val_loss


# Original training function (kept for compatibility)
def train_fold(train_loader, val_loader, model, epochs, lr=1e-3, hparams=None):
    model.train()
    # Only optimize parameters that require gradients (trainable parameters)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )

    log_dir = "/projectnb/ace-genetics/jueqiw/experiment/Autism_Brain_Development/experiments/tensorboard"
    if hparams and hasattr(hparams, "experiment_name"):
        log_dir = os.path.join(log_dir, f"{hparams.experiment_name}_fold")
    writer = SummaryWriter(log_dir=log_dir)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        total_train_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_age_loss = 0

        train_sample_for_logging = None
        for batch_idx, (imgs, ages) in enumerate(train_loader):
            imgs = imgs.to(device)
            ages = ages.to(device)

            optimizer.zero_grad()
            (
                recon,
                z_mean,
                z_logvar,
                pz_mean,
                pz_logvar,
                age_mean,
                age_logvar,
                age_pred,
            ) = model(imgs, ages)

            total_loss = vae_loss_fn(
                imgs,
                recon,
                z_mean,
                z_logvar,
                pz_mean,
                pz_logvar,
                age_mean,
                age_logvar,
                ages,
            )
            recon_loss = kl_loss = age_loss = total_loss  # For logging purposes

            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_age_loss += age_loss.item()

            # Store sample for logging
            if batch_idx == 0 and (epoch + 1) % 10 == 0:
                train_sample_for_logging = (
                    imgs[:4].detach().cpu(),
                    recon[:4].detach().cpu(),
                    ages[:4].detach().cpu(),
                )

        avg_train_loss = total_train_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        avg_age_loss = total_age_loss / len(train_loader)

        train_loss_history.append(avg_train_loss)

        # Log training losses
        writer.add_scalar("Loss/Train_Total", avg_train_loss, epoch)
        writer.add_scalar("Loss/Train_Reconstruction", avg_recon_loss, epoch)
        writer.add_scalar("Loss/Train_KL", avg_kl_loss, epoch)
        writer.add_scalar("Loss/Train_Age", avg_age_loss, epoch)

        # Validation
        model.eval()
        total_val_loss = 0
        sample_inputs = sample_reconstructions = sample_ages = None

        with torch.no_grad():
            for batch_idx, (imgs_val, ages_val) in enumerate(val_loader):
                imgs_val = imgs_val.to(device)
                ages_val = ages_val.to(device)

                (
                    recon_val,
                    z_mean_val,
                    z_logvar_val,
                    pz_mean_val,
                    pz_logvar_val,
                    age_mean_val,
                    age_logvar_val,
                    age_pred_val,
                ) = model(imgs_val, ages_val)

                val_loss = vae_loss_fn(
                    imgs_val,
                    recon_val,
                    z_mean_val,
                    z_logvar_val,
                    pz_mean_val,
                    pz_logvar_val,
                    age_mean_val,
                    age_logvar_val,
                    ages_val,
                )

                total_val_loss += val_loss.item()

                # Store samples for logging
                if batch_idx == 0 and (epoch + 1) % 10 == 0:
                    sample_inputs = imgs_val[:4].detach().cpu()
                    sample_reconstructions = recon_val[:4].detach().cpu()
                    sample_ages = ages_val[:4].detach().cpu()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

        # Log images every 10 epochs
        if (epoch + 1) % 10 == 0:
            if train_sample_for_logging is not None:
                sample_train_inputs, sample_train_recons, sample_train_ages = (
                    train_sample_for_logging
                )
                log_vae_images(
                    writer,
                    sample_train_inputs,
                    sample_train_recons,
                    sample_train_ages,
                    epoch,
                    prefix="Train",
                )

            if sample_inputs is not None:
                log_vae_images(
                    writer,
                    sample_inputs,
                    sample_reconstructions,
                    sample_ages,
                    epoch,
                    prefix="Val",
                )

            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
            )
            print(
                f"  Recon: {avg_recon_loss:.6f} | KL: {avg_kl_loss:.6f} | Age: {avg_age_loss:.6f}"
            )

        model.train()  # Switch back to training mode

    writer.close()
    return model


# Evaluation function
def evaluate(val_loader, model):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for imgs, ages in val_loader:
            imgs = imgs.to(device)
            ages = ages.to(device)
            # Get age prediction from VAE
            _, _, _, age_mean, _, _ = model(imgs, ages)
            preds.append(age_mean.cpu().numpy())
            targets.append(ages.cpu().numpy())
    preds = np.concatenate(preds).squeeze()
    targets = np.concatenate(targets).squeeze()
    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)
    return preds, targets, mse, r2


def main(hparams):
    # Load cached dataset directly
    cached_dataset = create_monai_dataset(cache_rate=0.1, num_workers=8)
    print(f"Loaded cached dataset with {len(cached_dataset)} samples")

    print("Splitting dataset into train/val/test (8:1:1)...")
    data_partitions = partition_dataset(
        data=list(range(len(cached_dataset))),
        ratios=[0.8, 0.1, 0.1],
        shuffle=True,
        seed=42,
    )

    train_indices, val_indices, test_indices = data_partitions
    print(f"Train samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")
    print(f"Test samples: {len(test_indices)}")

    train_dataset = MRIAgeDataset(cached_dataset, train_indices)
    val_dataset = MRIAgeDataset(cached_dataset, val_indices)
    test_dataset = MRIAgeDataset(cached_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Creating VAE model...")
    model = create_vae_model(
        hparams,
        pretrained_path=PRE_TRAINED_WEIGHTS
        / "all_model_monai_l1_autoencoder_diff_init_bs_64_heavily_weighted_loss_without_outside_brain_.pt",
    ).to(device)

    trained_model = train_fold_two_stage(
        train_loader, val_loader, model, epochs=hparams.n_epochs, hparams=hparams
    )

    print("Evaluating on validation set")
    val_preds, val_targets, val_mse, val_r2 = evaluate(val_loader, trained_model)
    print(f"Validation MSE: {val_mse:.4f}, R2: {val_r2:.4f}")

    print("Evaluating on test set...")
    test_preds, test_targets, test_mse, test_r2 = evaluate(test_loader, trained_model)
    print(f"Test MSE: {test_mse:.4f}, R2: {test_r2:.4f}")

    os.makedirs("torch_weights/vae_weights", exist_ok=True)
    torch.save(trained_model.state_dict(), "torch_weights/vae_weights/vae_final.pt")
    print("Model saved to torch_weights/vae_weights/vae_final.pt")

    print("Generating age-conditioned samples...")
    trained_model.eval()
    age_points = (
        torch.tensor([8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0], dtype=torch.float32)
        .unsqueeze(1)
        .to(device)
    )

    os.makedirs("torch_generations_vae/axial_generations", exist_ok=True)

    with torch.no_grad():
        # Generate age-dependent priors and sample from them
        pz_mean, pz_logvar = trained_model.age_to_prior(age_points)

        # Sample from the age-conditioned latent space
        z_samples = (
            trained_model.reparameterize(pz_mean, pz_logvar)
            if hasattr(trained_model, "reparameterize")
            else pz_mean
        )

        # Decode to images
        generated_images = trained_model.decode(z_samples)

    for i, age in enumerate(age_points.squeeze().cpu().numpy()):
        img = generated_images[i]
        save_image_tensor(
            img, f"torch_generations_vae/axial_generations/generated_age_{age:.1f}.png"
        )

    print("Generated images saved to torch_generations_vae/axial_generations/")


if __name__ == "__main__":
    parser = ArgumentParser(description="Trainer args", add_help=False)
    add_argument(parser)
    main(parser.parse_args())
    main()
