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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Resize,
    ScaleIntensityRange,
    ToTensor,
)
from monai.data import CacheDataset
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


# MONAI transforms for data preprocessing (no augmentation)
data_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize((192, 192)),
        ScaleIntensityRange(a_min=0.5, a_max=99.5, b_min=0.0, b_max=1.0, clip=True),
        ToTensor(),
    ]
)


# Normalize a single slice (kept for backward compatibility)
def normalize_slice(slice_data):
    lower, upper = np.percentile(slice_data, [0.5, 99.5])
    slice_clipped = np.clip(slice_data, lower, upper)
    normalized = (slice_clipped - lower) / (upper - lower + 1e-8)
    return normalized


# Old MRIDataset class removed - using MRIAgeDataset instead


class MRIAgeDataset(Dataset):
    """Custom dataset class that works with preprocessed tensors and ages"""

    def __init__(self, data_tensors, ages):
        self.data = data_tensors
        self.ages = ages

    def __len__(self):
        return len(self.ages)

    def __getitem__(self, idx):
        img = self.data[idx]
        age = torch.tensor([self.ages[idx]], dtype=torch.float32)
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
        transform=data_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
        progress=True,
    )

    return cached_ds


def extract_tensors_and_ages(cached_dataset):
    """Extract preprocessed tensors and ages from cached dataset"""
    print("Extracting preprocessed data from cache...")

    images = []
    ages = []

    for i in tqdm(range(len(cached_dataset)), desc="Extracting cached data"):
        sample = cached_dataset[i]
        images.append(sample["image"])
        ages.append(sample["age"])

    return torch.stack(images), np.array(ages)


def load_data_with_monai_cache(cache_rate=1.0, num_workers=4):
    """Load data using MONAI CacheDataset for optimal performance"""

    # Create cached dataset
    cached_ds = create_monai_dataset(cache_rate, num_workers)

    # Extract preprocessed tensors and ages
    data_tensors, ages = extract_tensors_and_ages(cached_ds)

    # Return in the format expected by the rest of the code
    return data_tensors.numpy(), ages


def load_data():
    """Main data loading function - uses MONAI CacheDataset for optimal speed"""
    return load_data_with_monai_cache()


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


# Training function per fold with TensorBoard logging
def train_fold(train_loader, val_loader, model, epochs, lr=1e-3, hparams=None):
    from models.vae import vae_loss_function

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Setup TensorBoard writer
    log_dir = "/projectnb/ace-genetics/jueqiw/experiment/Autism_Brain_Development/experiments/tensorboard"
    if hparams and hasattr(hparams, "experiment_name"):
        log_dir = os.path.join(log_dir, f"{hparams.experiment_name}_fold")
    writer = SummaryWriter(log_dir=log_dir)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        # Training
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

            # Compute VAE loss (using dummy targets for reconstruction since this is generative)
            total_loss, recon_loss, kl_loss, age_loss = vae_loss_function(
                recon,
                imgs,
                imgs,  # Use input as target for reconstruction
                z_mean,
                z_logvar,
                pz_mean,
                pz_logvar,
                age_mean,
                age_logvar,
                age_pred,
                ages,
            )

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

                val_loss, _, _, _ = vae_loss_function(
                    recon_val,
                    imgs_val,
                    imgs_val,
                    z_mean_val,
                    z_logvar_val,
                    pz_mean_val,
                    pz_logvar_val,
                    age_mean_val,
                    age_logvar_val,
                    age_pred_val,
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
    data, ages = load_data()
    print(
        "Data shape:", data.shape
    )  # (N, 1, H, W) - already has channel dimension from MONAI
    print("Ages shape:", ages.shape)

    # Data is already preprocessed by MONAI transforms
    ages = ages.astype(np.float32)

    # For cross-validation, we'll create datasets manually
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fake_strat = np.zeros(len(ages))
    preds_all = np.zeros(len(ages))

    for fold, (train_idx, val_idx) in enumerate(skf.split(data, fake_strat)):
        print(f"Fold {fold+1}")

        # Create datasets using the preprocessed data (no augmentation)
        train_dataset = MRIAgeDataset(data[train_idx], ages[train_idx])
        val_dataset = MRIAgeDataset(data[val_idx], ages[val_idx])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        model = create_vae_model(
            hparams,
            pretrained_path=PRE_TRAINED_WEIGHTS
            / "all_model_monai_l1_autoencoder_diff_init_bs_8_heavily_weighted_loss_without_outside_brain_.pt",
        ).to(device)

        model = train_fold(train_loader, val_loader, model, epochs=1, hparams=hparams)

        preds, targets, mse, r2 = evaluate(val_loader, model)
        print(f"Validation MSE: {mse:.4f}, R2: {r2:.4f}")
        preds_all[val_idx] = preds

        # Save VAE model weights
        os.makedirs("torch_weights/vae_weights", exist_ok=True)
        torch.save(model.state_dict(), f"torch_weights/vae_weights/vae_fold_{fold}.pt")
        break

    print("Cross-Validation MSE:", mean_squared_error(ages, preds_all))
    print("Cross-Validation R2:", r2_score(ages, preds_all))

    # Train on all data now to get final model
    print("Training on full dataset...")
    full_dataset = MRIAgeDataset(data, ages)
    full_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)
    model = create_vae_model(
        hparams,
        pretrained_path=PRE_TRAINED_WEIGHTS
        / "all_model_monai_l1_autoencoder_diff_init_bs_8_heavily_weighted_loss_without_outside_brain_.pt",
    ).to(device)

    # Train the final VAE model on full dataset
    final_model = train_fold(
        full_loader, full_loader, model, epochs=80, hparams=hparams
    )

    # Save final VAE model
    os.makedirs("torch_weights/vae_weights", exist_ok=True)
    torch.save(final_model.state_dict(), "torch_weights/vae_weights/vae_final.pt")

    # Generate samples from VAE given age points
    final_model.eval()
    age_points = (
        torch.tensor([8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0], dtype=torch.float32)
        .unsqueeze(1)
        .to(device)
    )

    os.makedirs("torch_generations_vae/axial_generations", exist_ok=True)

    with torch.no_grad():
        # Generate age-dependent priors and sample from them
        pz_mean, pz_logvar = final_model.age_to_prior(age_points)

        # Sample from the age-conditioned latent space
        z_samples = (
            final_model.reparameterize(pz_mean, pz_logvar)
            if hasattr(final_model, "reparameterize")
            else pz_mean
        )

        # Decode to images
        generated_images = final_model.decode(z_samples)

    for i, age in enumerate(age_points.squeeze().cpu().numpy()):
        img = generated_images[i]
        save_image_tensor(
            img, f"torch_generations_vae/axial_generations/generated_age_{age:.1f}.png"
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Trainer args", add_help=False)
    add_argument(parser)
    main(parser.parse_args())
    main()
