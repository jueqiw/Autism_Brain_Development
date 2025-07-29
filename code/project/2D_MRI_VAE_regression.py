import os
import glob
import sys
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
from monai.data import CacheDataset, partition_dataset, NibabelReader
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.add_argument import add_argument  # Assuming this is the correct import path
from utils.const import (
    ABIDE_I_2D_REGRESSION,
    ABIDE_II_2D_REGRESSION,
    ABIDE_PATH,
    PRE_TRAINED_WEIGHTS,
)
from models.vae import create_vae_model, save_vae_model, load_vae_model

torch.set_num_threads(8)  # or the number you request
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_one_hot_encoding(dx_group):
    """Create one-hot encoded conditional inputs for diagnosis

    Args:
        dx_group: 0=ASD, 1=Control

    Returns:
        one_hot: 1D numpy array with one-hot encoding (2 channels)
    """
    # Only 2 channels: ASD and Control
    one_hot = np.zeros(2, dtype=np.float32)

    # Diagnosis encoding: 0=ASD, 1=Control
    one_hot[dx_group] = 1.0

    return one_hot


def get_data_transforms():
    """Create MONAI transforms for data preprocessing with intensity normalization"""
    return Compose(
        [
            LoadImaged(
                keys=["image"],
                image_only=True,
                ensure_channel_first=True,  # (C, H, W) instead of (H, W)
                reader=NibabelReader(),
            ),
            Resized(keys=["image"], spatial_size=(192, 192)),
            ToTensord(keys=["image", "age", "dx_group", "dataset_num"]),
        ]
    )


class MRIAgeDataset(Dataset):
    """Custom dataset class that wraps MONAI CacheDataset for training with conditional input"""

    def __init__(self, cached_dataset, indices=None, use_conditional=True):
        self.cached_dataset = cached_dataset
        self.indices = (
            indices if indices is not None else list(range(len(cached_dataset)))
        )
        self.use_conditional = use_conditional

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the actual index from the subset
        actual_idx = self.indices[idx]
        sample = self.cached_dataset[actual_idx]

        img = sample["image"]  # Already preprocessed tensor (1, H, W)

        # Normalize age to [-1, 1] range to match tanh output (assuming ages are 0-21)
        age_normalized = (sample["age"] / 21.0) * 2.0 - 1.0  # [0,1] -> [-1,1]
        age = torch.tensor([age_normalized], dtype=torch.float32)

        if self.use_conditional:
            # Create one-hot conditional input
            dx_group = int(sample["dx_group"])  # 0=ASD, 1=Control

            # Get one-hot encoding
            one_hot = create_one_hot_encoding(dx_group)

            # Convert to tensor and create channel maps
            one_hot_tensor = torch.from_numpy(one_hot)  # Shape: (2,)
            H, W = img.shape[1], img.shape[2]

            # Create one-hot channel maps by broadcasting to image size
            conditional_channels = (
                one_hot_tensor.unsqueeze(-1).unsqueeze(-1).expand(-1, H, W)
            )  # Shape: (2, H, W)

            # Concatenate original image with conditional channels
            # Result: (1 + 2, H, W) = (3, H, W)
            img_with_conditional = torch.cat([img, conditional_channels], dim=0)

            return img_with_conditional, age
        else:
            return img, age


# Loss function combining reconstruction, KL, and label (age) loss
def vae_loss_fn(
    x,
    x_recon,
    z_mean,
    z_log_var,
    pz_mean,
    pz_log_var,
    r_mean,
    r_log_var,
    r,
    recon_weight=1.0,
    kl_weight=0.01,
    age_weight=10.0,
):

    # save the target image
    recon_loss = F.l1_loss(x_recon, x[:, 0, :, :].unsqueeze(1), reduction="mean")

    # KL divergence between posterior z and prior pz
    kl_loss = (
        1
        + z_log_var
        - pz_log_var
        - ((z_mean - pz_mean).pow(2) / pz_log_var.exp())
        - (z_log_var.exp() / pz_log_var.exp())
    )
    kl_loss = -0.5 * kl_loss.sum(dim=1).mean()

    # Simple age regression loss (MSE) - more stable than probabilistic loss
    age_loss = F.mse_loss(r_mean, r, reduction="mean")

    # Combined weighted loss
    total_loss = recon_weight * recon_loss + kl_weight * kl_loss + age_weight * age_loss

    return total_loss, recon_loss, kl_loss, age_loss


# Augmentation functions removed - no data augmentation used


# Loading all data (same logic as TF version)
def get_phenotypic_data(sub_id, dataset_num):
    """Get age, diagnosis group, and site info for a subject"""
    sub_id = int(sub_id)
    if dataset_num == 1:
        df = pd.read_csv(ABIDE_PATH / "Phenotypic_V1_0b.csv")
        row = df[(df["SUB_ID"] == sub_id)]
        if row.empty:
            return None
        age = row["AGE_AT_SCAN"].values[0]
        if age > 21:
            return None
        dx_group = (
            int(row["DX_GROUP"].values[0]) - 1
        )  # Convert to 0-indexed (0=ASD, 1=Control)
        site_id = row["SITE_ID"].values[0] if "SITE_ID" in row.columns else "Unknown"
        return age, dx_group, site_id
    elif dataset_num == 2:
        df = pd.read_csv(
            ABIDE_PATH / "ABIDEII_Composite_Phenotypic.csv", encoding="cp1252"
        )
        row = df[(df["SUB_ID"] == sub_id)]
        if row.empty:
            return None
        age = row["AGE_AT_SCAN "].values[0]  # note extra space
        if age > 21:
            return None
        dx_group = (
            int(row["DX_GROUP"].values[0]) - 1
        )  # Convert to 0-indexed (0=ASD, 1=Control)
        site_id = row["SITE_ID"].values[0] if "SITE_ID" in row.columns else "Unknown"
        return age, dx_group, site_id
    return None


def get_ages(sub_id, dataset_num):
    """Backward compatibility function"""
    result = get_phenotypic_data(sub_id, dataset_num)
    return result[0] if result is not None else None


def prepare_data_dicts():
    """Prepare data dictionaries for MONAI CacheDataset with phenotypic data"""
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
            if filename.endswith(".npy"):
                subject_id = filename[2:7]
                dataset_num = 1 if "ABIDE_I_2D" in str(folder_path) else 2
                phenotypic_data = get_phenotypic_data(subject_id, dataset_num)

                if phenotypic_data is None:
                    continue

                age, dx_group, site_id = phenotypic_data
                if age < 21:
                    img_path = str(folder_path / filename)

                    data_dicts.append(
                        {
                            "image": img_path,
                            "age": float(age),
                            "dx_group": int(dx_group),  # 0=ASD, 1=Control
                            "dataset_num": int(dataset_num),  # 1=ABIDE-I, 2=ABIDE-II
                            "site_id": site_id,
                            "subject_id": subject_id,
                        }
                    )

    print(f"Found {len(data_dicts)} valid samples")
    return data_dicts


def create_monai_dataset(cache_rate=1.0, num_workers=4):
    """Create MONAI CacheDataset for data loading"""

    data_dicts = prepare_data_dicts()

    if len(data_dicts) == 0:
        raise ValueError("No valid data found!")

    print("Creating MONAI CacheDataset...")
    cached_ds = CacheDataset(
        data=data_dicts,
        transform=get_data_transforms(),
        cache_rate=cache_rate,
        num_workers=num_workers,
        progress=True,
    )

    return cached_ds


def log_vae_images(writer, inputs, reconstructions, ages, epoch, prefix=""):
    """
    Log VAE input and reconstruction images to TensorBoard with age info
    """
    sample_idx = 0

    # Get first sample - handle 3-channel input by using only first channel
    if inputs.shape[1] == 3:
        input_img = inputs[
            sample_idx : sample_idx + 1, 0:1, :, :
        ]  # Take only first channel (brain image)
    else:
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

    # Simplified title with just epoch number
    title = f"{prefix}_Epoch_{epoch+1}"
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

    stage_transition_epoch = getattr(hparams, "stage_transition_epoch", 40)
    warmup_epochs = min(stage_transition_epoch, epochs)
    finetune_epochs = epochs - warmup_epochs
    stage_epochs = max(1, finetune_epochs // 3) if finetune_epochs > 0 else 1

    print(
        f"Two-stage training: {warmup_epochs} warm-up + {finetune_epochs} fine-tuning epochs"
    )
    print(
        f"Stage transition configured for epoch {stage_transition_epoch}, actual transition at epoch {warmup_epochs}"
    )

    total_train_history = []
    total_val_history = []

    print("Stage 1: Warm-up with frozen encoder...")
    model.freeze_pretrained_encoder()
    model.freeze_pretrained_decoder()

    param_groups = model.get_parameter_groups(
        new_lr=lr, pretrained_lr=lr / 10, weight_decay=1e-4
    )
    optimizer = torch.optim.Adam(param_groups)

    for epoch in range(warmup_epochs):
        (
            train_loss,
            val_loss,
            avg_recon_loss,
            avg_kl_loss,
            avg_age_loss,
            avg_perceptual_loss,
        ) = train_epoch(
            train_loader, val_loader, model, optimizer, writer, epoch, "warmup", hparams
        )
        total_train_history.append(train_loss)
        total_val_history.append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(
                f"Warm-up Epoch {epoch+1}/{warmup_epochs} - Total: {train_loss:.6f} | Val: {val_loss:.6f}"
            )
            print(
                f"  Loss components - Recon: {avg_recon_loss:.6f}, KL: {avg_kl_loss:.6f}, Age: {avg_age_loss:.6f}"
            )

        # Early transition check - advance to next stage at configured epoch
        if epoch + 1 >= stage_transition_epoch:
            print(f"Advancing to fine-tuning stage at epoch {epoch + 1}")
            break

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
            train_loss, val_loss, _, _, _ = train_epoch(
                train_loader,
                val_loader,
                model,
                optimizer,
                writer,
                epoch,
                f"finetune_stage_{stage}",
                hparams,
            )
            total_train_history.append(train_loss)
            total_val_history.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Fine-tune Stage {stage} Epoch {epoch+1-stage_start_epoch+1}/{stage_end_epoch-stage_start_epoch} - Train: {train_loss:.6f} | Val: {val_loss:.6f}"
                )

    writer.close()
    return model


def train_epoch(
    train_loader, val_loader, model, optimizer, writer, epoch, stage_name, hparams=None
):
    """Single epoch training and validation"""
    model.train()

    # Training phase
    total_train_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_age_loss = 0
    total_perceptual_loss = 0

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

        # Compute VAE loss using local vae_loss_fn with configurable weights
        recon_weight = getattr(hparams, "recon_weight", 1.0)
        base_kl_weight = getattr(hparams, "kl_weight", 0.01)
        age_weight = getattr(hparams, "age_weight", 10.0)

        # KL annealing: gradually increase KL weight to prevent rapid increase
        max_kl_weight = 0.05  # Target KL weight (5x higher than current)
        annealing_epochs = getattr(hparams, "kl_annealing_epochs", 100)
        kl_weight = min(
            max_kl_weight,
            base_kl_weight
            + (max_kl_weight - base_kl_weight) * (epoch / annealing_epochs),
        )

        # Use perceptual loss for sharper images
        perceptual_weight = getattr(hparams, "perceptual_weight", 0.01)

        loss_result = vae_loss_fn(
            imgs,
            recon,
            z_mean,
            z_logvar,
            pz_mean,
            pz_logvar,
            age_mean,
            age_logvar,
            ages,
            recon_weight=recon_weight,
            kl_weight=kl_weight,
            age_weight=age_weight,
            perceptual_weight=perceptual_weight,
        )

        # Handle different return formats (backward compatibility)
        if len(loss_result) == 5:
            total_loss, recon_loss, kl_loss, age_loss, perceptual_loss = loss_result
        else:
            total_loss, recon_loss, kl_loss, age_loss = loss_result
            perceptual_loss = 0.0

        total_loss.backward()
        optimizer.step()

        total_train_loss += total_loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        total_age_loss += age_loss.item()
        total_perceptual_loss += (
            perceptual_loss
            if isinstance(perceptual_loss, (int, float))
            else perceptual_loss.item()
        )

        # Store sample for logging
        if batch_idx == 0 and (epoch + 1) % 10 == 0:
            train_sample_for_logging = (
                imgs[:4].detach().cpu(),
                recon[:4].detach().cpu(),
                ages[:4].detach().cpu(),
            )

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_recon = total_recon_loss / len(train_loader)
    avg_train_kl = total_kl_loss / len(train_loader)
    avg_train_age = total_age_loss / len(train_loader)
    avg_train_perceptual = total_perceptual_loss / len(train_loader)

    # Validation phase
    model.eval()
    total_val_loss = 0
    total_val_recon = 0
    total_val_kl = 0
    total_val_age = 0
    total_val_perceptual = 0
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

            val_loss_result = vae_loss_fn(
                imgs_val,
                recon_val,
                z_mean_val,
                z_logvar_val,
                pz_mean_val,
                pz_logvar_val,
                age_mean_val,
                age_logvar_val,
                ages_val,
                perceptual_weight=perceptual_weight,
            )

            # Handle different return formats (backward compatibility)
            if len(val_loss_result) == 5:
                val_loss, val_recon, val_kl, val_age, val_perceptual = val_loss_result
            else:
                val_loss, val_recon, val_kl, val_age = val_loss_result
                val_perceptual = 0.0

            total_val_loss += val_loss.item()
            total_val_recon += val_recon.item()
            total_val_kl += val_kl.item()
            total_val_age += val_age.item()
            total_val_perceptual += (
                val_perceptual
                if isinstance(val_perceptual, (int, float))
                else val_perceptual.item()
            )

            # Store samples for logging
            if batch_idx == 0 and (epoch + 1) % 10 == 0:
                sample_inputs = imgs_val[:4].detach().cpu()
                sample_reconstructions = recon_val[:4].detach().cpu()
                sample_ages = ages_val[:4].detach().cpu()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_recon = total_val_recon / len(val_loader)
    avg_val_kl = total_val_kl / len(val_loader)
    avg_val_age = total_val_age / len(val_loader)
    avg_val_perceptual = total_val_perceptual / len(val_loader)

    # Log metrics to TensorBoard - separate training and validation plots
    # Training losses
    writer.add_scalar("train_loss/total", avg_train_loss, epoch)
    writer.add_scalar("train_loss/reconstruction", avg_train_recon, epoch)
    writer.add_scalar("train_loss/kl_divergence", avg_train_kl, epoch)
    writer.add_scalar("train_loss/age_regression", avg_train_age, epoch)
    writer.add_scalar("train_loss/perceptual", avg_train_perceptual, epoch)

    # Validation losses
    writer.add_scalar("val_loss/total", avg_val_loss, epoch)
    writer.add_scalar("val_loss/reconstruction", avg_val_recon, epoch)
    writer.add_scalar("val_loss/kl_divergence", avg_val_kl, epoch)
    writer.add_scalar("val_loss/age_regression", avg_val_age, epoch)
    writer.add_scalar("val_loss/perceptual", avg_val_perceptual, epoch)

    # Log images every 10 epochs - simplified titles with just epoch
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
                prefix="Validation",
            )

    return (
        avg_train_loss,
        avg_val_loss,
        avg_train_recon,
        avg_train_kl,
        avg_train_age,
        avg_train_perceptual,
    )


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
            (
                recon_val,
                z_mean_val,
                z_logvar_val,
                pz_mean_val,
                pz_logvar_val,
                age_mean_val,
                age_logvar_val,
                age_pred_val,
            ) = model(imgs, ages)
            preds.append(age_pred_val.cpu().numpy())
            targets.append(ages.cpu().numpy())
    preds = np.concatenate(preds).squeeze()
    targets = np.concatenate(targets).squeeze()

    # Denormalize predictions and targets back to original age scale
    # Convert from [-1,1] back to [0,21]
    preds_denorm = (preds + 1.0) / 2.0 * 21.0
    targets_denorm = (targets + 1.0) / 2.0 * 21.0

    mse = mean_squared_error(targets_denorm, preds_denorm)
    r2 = r2_score(targets_denorm, preds_denorm)

    print(
        f"Age range - Predicted: [{preds_denorm.min():.1f}, {preds_denorm.max():.1f}], "
        f"Actual: [{targets_denorm.min():.1f}, {targets_denorm.max():.1f}]"
    )

    return preds_denorm, targets_denorm, mse, r2


def main(hparams):
    # Load cached dataset directly
    cached_dataset = create_monai_dataset(cache_rate=1, num_workers=8)
    print(f"Loaded cached dataset with {len(cached_dataset)} samples")
    print("Splitting dataset by subjects into train/val/test (8:1:1)...")

    # Get all unique subjects and their sample indices
    subject_to_indices = {}
    for idx, sample in enumerate(cached_dataset.data):
        subject_id = sample["subject_id"]
        if subject_id not in subject_to_indices:
            subject_to_indices[subject_id] = []
        subject_to_indices[subject_id].append(idx)

    unique_subjects = list(subject_to_indices.keys())
    np.random.seed(42)
    np.random.shuffle(unique_subjects)

    n_subjects = len(unique_subjects)
    train_split = int(0.8 * n_subjects)
    val_split = int(0.9 * n_subjects)

    train_subjects = unique_subjects[:train_split]
    val_subjects = unique_subjects[train_split:val_split]
    test_subjects = unique_subjects[val_split:]

    # Get sample indices for each split
    train_indices = []
    val_indices = []
    test_indices = []

    for subject in train_subjects:
        train_indices.extend(subject_to_indices[subject])
    for subject in val_subjects:
        val_indices.extend(subject_to_indices[subject])
    for subject in test_subjects:
        test_indices.extend(subject_to_indices[subject])

    print(
        f"Subjects - Train: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)}"
    )
    print(
        f"Samples - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}"
    )

    train_dataset = MRIAgeDataset(cached_dataset, train_indices, use_conditional=True)
    val_dataset = MRIAgeDataset(cached_dataset, val_indices, use_conditional=True)
    test_dataset = MRIAgeDataset(cached_dataset, test_indices, use_conditional=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = create_vae_model(
        hparams,
        pretrained_path=PRE_TRAINED_WEIGHTS
        / "all_model_monai_l1_autoencoder_diff_init_bs_64_heavily_weighted_loss_without_outside_brain_.pt",
    ).to(device)

    trained_model = train_fold_two_stage(
        train_loader, val_loader, model, epochs=hparams.n_epochs, hparams=hparams
    )

    val_preds, val_targets, val_mse, val_r2 = evaluate(val_loader, trained_model)
    print(f"Validation MSE: {val_mse:.4f}, R2: {val_r2:.4f}")

    print("Evaluating on test set...")
    test_preds, test_targets, test_mse, test_r2 = evaluate(test_loader, trained_model)
    print(f"Test MSE: {test_mse:.4f}, R2: {test_r2:.4f}")

    result_folder = Path(
        "/projectnb/ace-genetics/jueqiw/experiment/Autism_Brain_Development/experiments/vae_weight"
    )
    # Save the trained VAE model using the custom save function
    os.makedirs(str(result_folder), exist_ok=True)
    save_vae_model(
        trained_model,
        str(result_folder / f"vae_{hparams.experiment_name}.pt"),
        hparams=hparams,
        epoch=hparams.n_epochs,
    )

    # trained_model.eval()
    # age_points = (
    #     torch.tensor([8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0], dtype=torch.float32)
    #     .unsqueeze(1)
    #     .to(device)
    # )

    # os.makedirs("torch_generations_vae/axial_generations", exist_ok=True)

    # with torch.no_grad():
    #     # Generate age-dependent priors and sample from them
    #     pz_mean, pz_logvar = trained_model.age_to_prior(age_points)

    #     # Sample from the age-conditioned latent space
    #     z_samples = (
    #         trained_model.reparameterize(pz_mean, pz_logvar)
    #         if hasattr(trained_model, "reparameterize")
    #         else pz_mean
    #     )

    #     # Decode to images
    #     generated_images = trained_model.decode(z_samples)

    # for i, age in enumerate(age_points.squeeze().cpu().numpy()):
    #     img = generated_images[i]
    #     save_image_tensor(
    #         img, f"torch_generations_vae/axial_generations/generated_age_{age:.1f}.png"
    #     )

    # print("Generated images saved to torch_generations_vae/axial_generations/")

    # Test the model loading functionality
    test_model_loading(
        str(result_folder / f"vae_{hparams.experiment_name}.pt"), test_loader
    )


def test_model_loading(model_path, test_loader):
    """Test the model loading functionality"""
    print(f"\n=== Testing Model Loading from {model_path} ===")

    try:
        # Load the saved model
        loaded_model, checkpoint_info = load_vae_model(model_path, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(
            "This might be an old-style model save. Please re-save using save_vae_model()."
        )
        return None

    # Set to evaluation mode
    loaded_model.eval()

    # Test on a small batch
    with torch.no_grad():
        for imgs, ages in test_loader:
            imgs = imgs.to(device)
            ages = ages.to(device)

            # Forward pass with loaded model
            output = loaded_model(imgs, ages)
            reconstruction = output[0]

    print("=== Model Loading Test Complete ===\n")
    return loaded_model


if __name__ == "__main__":
    parser = ArgumentParser(description="Trainer args", add_help=False)
    add_argument(parser)
    main(parser.parse_args())
