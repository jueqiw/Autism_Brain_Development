import os
import glob
import sys
import importlib
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import pickle
from pathlib import Path
import datetime

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
    Resized,
    ScaleIntensityRanged,
    ToTensord,
)
from monai.data import CacheDataset, partition_dataset, NibabelReader
from monai.losses import PerceptualLoss
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.add_argument import add_argument
from utils.const import (
    ABIDE_I_2D_REGRESSION,
    ABIDE_II_2D_REGRESSION,
    ABIDE_PATH,
    PRE_TRAINED_WEIGHTS,
)

# Aggressive module cache clearing to force fresh model definitions
import sys
import gc

# Remove any cached modules related to models
modules_to_remove = [k for k in sys.modules.keys() if k.startswith("models.")]
for module_name in modules_to_remove:
    if module_name in sys.modules:
        del sys.modules[module_name]

# Force garbage collection to clear any remaining references
gc.collect()

# Clear CUDA cache if available
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Fresh import after cache clearing
import models.gan_generator
import models.discriminator_new

from models.gan_generator import create_gan_generator, MODEL_VERSION as GEN_VERSION
from models.discriminator_new import create_discriminator, MODEL_VERSION as DISC_VERSION

torch.set_num_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Set device with memory optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(
        f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )


def create_one_hot_encoding(dx_group):
    """Create conditional inputs for diagnosis

    Args:
        dx_group: 0=ASD, 1=Control

    Returns:
        condition: 1D numpy array with diagnosis value (1 channel)
    """
    # Use single value instead of one-hot: 0.0 for ASD, 1.0 for Control
    condition = np.array([float(dx_group)], dtype=np.float32)

    return condition


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
            # Scale intensity to [-1, 1] range
            ScaleIntensityRanged(
                keys=["image"],
                a_min=0.0,
                a_max=1.0,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
            ),
            ToTensord(keys=["image", "age", "dx_group", "dataset_num"]),
        ]
    )


class MRIGANDataset(Dataset):
    """Custom dataset class for GAN training with conditional input (age and ASD vectors)"""

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

        img = sample["image"]  # Already preprocessed tensor (1, H, W)

        age = int(sample["age"])
        age_vector = torch.zeros(20, dtype=torch.float32)  # 25-dimensional vector
        if age > 0:
            age_vector[: min(age, 20)] = 1.0  # Set first 'age' elements to 1

        # Create ASD/diagnosis vector (1-dimensional)
        dx_group = int(sample["dx_group"])  # 0=ASD, 1=Control
        asd_vector = torch.tensor(
            [float(dx_group)], dtype=torch.float32
        )  # Single value: 0.0 or 1.0

        return img, age_vector, asd_vector


def adversarial_loss(pred, target):
    """Binary cross-entropy loss for GAN training - handles both single and patch outputs"""
    # If pred has spatial dimensions (patch discriminator), expand target to match
    if len(pred.shape) > 2:  # Shape: (batch_size, 1, height, width)
        # Reshape target from (batch_size, 1) to (batch_size, 1, 1, 1) then expand
        target = target.unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1, 1)
        target = target.expand_as(pred)  # (batch_size, 1, height, width)
    return F.binary_cross_entropy_with_logits(pred, target)


def gradient_penalty(
    discriminator, real_data, fake_data, age_vec, asd_vec, device, lambda_gp=10
):
    """Calculate gradient penalty for WGAN-GP"""
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)

    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    d_interpolated = discriminator(interpolated, age_vec, asd_vec)

    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Calculate gradient penalty - flatten all dimensions except batch
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def generate_epoch_result_plots(
    generator,
    real_imgs,
    age_vecs,
    asd_vecs,
    epoch,
    writer,
    device,
    hparams,
    num_samples=4,
):
    """Generate comprehensive result plots for periodic logging during training"""
    generator.eval()

    with torch.no_grad():
        # Ensure we have the right number of samples
        real_imgs = real_imgs[:num_samples]
        age_vecs = age_vecs[:num_samples]
        asd_vecs = asd_vecs[:num_samples]

        # Create shifted age vectors for age manipulation
        shifted_age_vecs = age_vecs.clone()
        new_age_vecs = age_vecs.clone()
        for i in range(num_samples):
            current_age = age_vecs[i].sum().item()
            age_shift = torch.randint(1, 4, (1,)).item()  # Random shift 1-3 years
            new_age = max(1, min(20, current_age + age_shift))
            shifted_age_vecs[i] = torch.zeros_like(age_vecs[i])
            new_age_vecs[i] = torch.zeros_like(age_vecs[i])
            if new_age > 0:
                shifted_age_vecs[i][: min(int(age_shift), 20)] = 1.0
                new_age_vecs[i][: min(int(new_age), 20)] = 1.0

        fake_imgs = generator(real_imgs, shifted_age_vecs, asd_vecs)
        fig, axes = plt.subplots(3, num_samples, figsize=(2 * num_samples, 6))

        for i in range(num_samples):
            real_img = real_imgs[i, 0].cpu().numpy()
            age = age_vecs[i].sum().item()
            new_age = new_age_vecs[i].sum().item()
            dx = "ASD" if asd_vecs[i, 0].item() > 0.5 else "Control"

            axes[0, i].imshow(real_img, cmap="gray")
            axes[0, i].set_title(f"Real\nAge:{age:.0f}, {dx}")
            axes[0, i].axis("off")

            fake_img = fake_imgs[i, 0].cpu().numpy()
            axes[1, i].imshow(fake_img, cmap="gray")
            axes[1, i].set_title(f"Generated\nAge:{new_age:.0f}, {dx}")
            axes[1, i].axis("off")

            diff_img = fake_img - real_img
            im = axes[2, i].imshow(diff_img, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
            axes[2, i].set_title(f"Difference\n{age:.0f}→{new_age:.0f}")
            axes[2, i].axis("off")
            plt.colorbar(im, ax=axes[2, i], shrink=0.6)

        plt.suptitle(f"GAN Results - Epoch {epoch}", fontsize=16, fontweight="bold")
        plt.tight_layout()

        # Log the comprehensive figure to TensorBoard
        writer.add_figure(
            f"Epoch_Results/Comprehensive_Comparison_Epoch_{epoch:03d}", fig, epoch
        )
        plt.close(fig)

        # Log individual images with detailed titles
        for i in range(num_samples):
            age = age_vecs[i].sum().item()
            new_age = new_age_vecs[i].sum().item()
            dx = "ASD" if asd_vecs[i, 0].item() > 0.5 else "Control"

            # Log real image
            real_img_tensor = (
                real_imgs[i].squeeze(0) if real_imgs[i].dim() == 4 else real_imgs[i]
            )
            try:
                writer.add_image(
                    f"Epoch_{epoch:03d}_Real/Sample_{i+1:02d}_Age_{age:.0f}_{dx}",
                    real_img_tensor,
                    epoch,
                    dataformats="CHW",
                )
            except Exception as e:
                print(f"Error logging epoch {epoch} real image {i}: {e}")

            # Log generated image
            fake_img_tensor = (
                fake_imgs[i].squeeze(0) if fake_imgs[i].dim() == 4 else fake_imgs[i]
            )
            try:
                writer.add_image(
                    f"Epoch_{epoch:03d}_Generated/Sample_{i+1:02d}_Age_{age:.0f}_{dx}",
                    fake_img_tensor,
                    epoch,
                    dataformats="CHW",
                )
            except Exception as e:
                print(f"Error logging epoch {epoch} fake image {i}: {e}")

        print(f"Epoch {epoch} result plots logged to TensorBoard")

    generator.train()


def train_gan(
    train_loader,
    generator,
    discriminator,
    epochs,
    lr_g=2e-4,
    lr_d=1e-4,
    hparams=None,
    perceptual_loss_fn=None,
):
    """Train GAN with age and ASD conditioning - Memory optimized version with discriminator balancing"""

    # Option 4: Simple Fixed LR - Conservative learning rates for medical imaging
    lr_g_fixed = 1e-4  # Slightly lower for generator stability
    lr_d_fixed = 2e-4  # Slightly higher for discriminator

    # Use RMSprop for better stability with fixed learning rates
    optimizer_g = torch.optim.RMSprop(generator.parameters(), lr=lr_g_fixed)
    optimizer_d = torch.optim.RMSprop(discriminator.parameters(), lr=lr_d_fixed)

    print(f"Using Simple Fixed LR: Generator={lr_g_fixed}, Discriminator={lr_d_fixed}")
    print("No learning rate scheduling - RMSprop optimizer for stability")

    log_dir = "/projectnb/ace-genetics/jueqiw/experiment/Autism_Brain_Development/experiments/tensorboard"
    if hparams and hasattr(hparams, "experiment_name"):
        log_dir = os.path.join(log_dir, f"{hparams.experiment_name}_gan")
    writer = SummaryWriter(log_dir=log_dir)

    generator.train()
    discriminator.train()

    adversarial_weight = getattr(hparams, "adversarial_weight", 1.0)
    content_weight = getattr(hparams, "content_weight", 5.0)
    perceptual_weight = getattr(hparams, "perceptual_weight", 0.1)
    gradient_penalty_weight = getattr(hparams, "gradient_penalty_weight", 10.0)
    use_wgan_gp = getattr(hparams, "use_wgan_gp", False)

    d_train_freq = getattr(hparams, "d_train_freq", 1)
    g_train_freq = getattr(hparams, "g_train_freq", 1)  # Train G every iteration

    # Moderate regularization
    label_smoothing = getattr(hparams, "label_smoothing", 0.1)
    d_loss_threshold = getattr(hparams, "d_loss_threshold", 0.5)

    print(f"Training GAN for {epochs} epochs")
    print(
        f"Fixed Learning Rates - Generator: {lr_g_fixed}, Discriminator: {lr_d_fixed}"
    )
    print(
        f"Loss weights - Adversarial: {adversarial_weight}, Content: {content_weight}, Perceptual: {perceptual_weight}"
    )
    print(f"Training frequencies - D: every {d_train_freq}, G: every {g_train_freq}")
    print(f"Regularization - Label smoothing: {label_smoothing}")
    print(f"Adaptive training - D loss threshold: {d_loss_threshold}")

    scaler_g = torch.cuda.amp.GradScaler()
    scaler_d = torch.cuda.amp.GradScaler()
    print("Using mixed precision training for memory efficiency")

    for epoch in range(epochs):
        g_losses = []
        d_losses = []
        content_losses = []

        for batch_idx, (real_imgs, age_vecs, asd_vecs) in enumerate(train_loader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device, non_blocking=True)
            real_age_vecs = age_vecs.to(device, non_blocking=True)
            asd_vecs = asd_vecs.to(device, non_blocking=True)

            shifted_age_vecs = real_age_vecs.clone()
            new_age_vecs = real_age_vecs.clone()
            if getattr(hparams, "enable_age_manipulation", True):
                manipulation_mask = torch.rand(batch_size) < 0.5
                for i in range(batch_size):
                    if manipulation_mask[i]:
                        current_age = real_age_vecs[i].sum().item()
                        age_shift = torch.randint(1, 4, (1,)).item()
                        if current_age + age_shift > 20:
                            age_shift = 20 - current_age
                        new_age = max(1, min(20, current_age + age_shift))
                        shifted_age_vecs[i] = torch.zeros_like(real_age_vecs[i])
                        new_age_vecs[i] = torch.zeros_like(real_age_vecs[i])
                        if new_age > 0:
                            shifted_age_vecs[i][: min(int(age_shift), 20)] = 1.0
                            new_age_vecs[i][: min(int(new_age), 20)] = 1.0

            real_labels = torch.ones(batch_size, 1, device=device) * (
                1.0 - label_smoothing
            )
            fake_labels = torch.zeros(batch_size, 1, device=device)
            should_train_discriminator = batch_idx % d_train_freq == 0

            if should_train_discriminator:
                optimizer_d.zero_grad()
                with torch.cuda.amp.autocast():

                    real_output, age_output = discriminator(real_imgs, asd_vecs)
                    fake_imgs = generator(
                        real_imgs, shifted_age_vecs, asd_vecs
                    ).detach()
                    fake_output, real_fake_age_output = discriminator(
                        fake_imgs, asd_vecs
                    )
                    d_loss_real = adversarial_loss(real_output, real_labels)
                    d_loss_fake = adversarial_loss(fake_output, fake_labels)

                    true_ages = torch.tensor(
                        [real_age_vecs[i].sum().item() for i in range(batch_size)],
                        dtype=torch.float32,
                        device=device,
                    )
                    fake_ages = torch.tensor(
                        [new_age_vecs[i].sum().item() for i in range(batch_size)],
                        dtype=torch.float32,
                        device=device,
                    )
                    age_loss = F.mse_loss(age_output.squeeze(), true_ages / 20.0)
                    age_loss_1 = F.mse_loss(
                        real_fake_age_output.squeeze(), fake_ages / 20.0
                    )

                    d_loss = (d_loss_real + d_loss_fake) / 2 + (age_loss + age_loss_1)

                current_d_loss = d_loss.item()
                if current_d_loss > d_loss_threshold:
                    scaler_d.scale(d_loss).backward()
                    # Clip gradients to prevent explosion
                    scaler_d.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(
                        discriminator.parameters(), max_norm=1.0
                    )
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    pass

                d_real_loss_item = d_loss_real.item()
                d_fake_loss_item = d_loss_fake.item()
            else:
                # Skip discriminator training but still generate fake images for metrics
                with torch.no_grad():
                    fake_imgs = generator(real_imgs, shifted_age_vecs, asd_vecs)
                    real_output = discriminator(real_imgs, real_age_vecs, asd_vecs)
                    fake_output = discriminator(fake_imgs, new_age_vecs, asd_vecs)
                    d_loss_real = adversarial_loss(real_output, real_labels)
                    d_loss_fake = adversarial_loss(fake_output, fake_labels)
                    current_d_loss = ((d_loss_real + d_loss_fake) / 2).item()
                    d_real_loss_item = d_loss_real.item()
                    d_fake_loss_item = d_loss_fake.item()

            if batch_idx % g_train_freq == 0:
                optimizer_g.zero_grad()

                with torch.cuda.amp.autocast():
                    # Generate NEW fake images using SHIFTED ages for generator training
                    fake_imgs_gen = generator(real_imgs, shifted_age_vecs, asd_vecs)
                    fake_output_gen = discriminator(
                        fake_imgs_gen, new_age_vecs, asd_vecs
                    )
                    g_loss_adv = adversarial_loss(fake_output_gen, real_labels)

                    # Hybrid content loss for sharper images
                    g_loss_l1 = F.l1_loss(fake_imgs_gen, real_imgs)
                    g_loss_l2 = F.mse_loss(fake_imgs_gen, real_imgs)
                    g_loss_content = 0.8 * g_loss_l1 + 0.2 * g_loss_l2
                    perceptual_loss = perceptual_loss_fn(fake_imgs_gen, real_imgs)

                    # Simple generator loss - back to basics
                    g_loss = (
                        adversarial_weight * g_loss_adv
                        + content_weight * g_loss_content
                        + perceptual_weight * perceptual_loss
                    )

                # Backward pass with gradient scaling
                scaler_g.scale(g_loss).backward()
                # Clip gradients to prevent explosion
                scaler_g.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                scaler_g.step(optimizer_g)
                scaler_g.update()

                current_g_loss = g_loss.item()
                current_content_loss = g_loss_content.item()
            else:
                with torch.no_grad():
                    fake_imgs_gen = generator(real_imgs, shifted_age_vecs, asd_vecs)
                    fake_output_gen = discriminator(
                        fake_imgs_gen, new_age_vecs, asd_vecs
                    )
                    g_loss_adv = adversarial_loss(fake_output_gen, real_labels)
                    g_loss_l1 = F.l1_loss(fake_imgs_gen, real_imgs)
                    g_loss_l2 = F.mse_loss(fake_imgs_gen, real_imgs)
                    g_loss_content = 0.8 * g_loss_l1 + 0.2 * g_loss_l2
                    current_g_loss = (
                        adversarial_weight * g_loss_adv
                        + content_weight * g_loss_content
                    ).item()
                    current_content_loss = g_loss_content.item()
                    current_perceptual_loss = perceptual_loss.item()

            # Check for NaN losses and restart if needed
            if torch.isnan(torch.tensor(current_d_loss)) or torch.isnan(
                torch.tensor(current_g_loss)
            ):
                print(
                    f"NaN detected at batch {batch_idx}! D_loss: {current_d_loss}, G_loss: {current_g_loss}"
                )
                print("Stopping training to prevent further instability.")
                break

            g_losses.append(current_g_loss)
            d_losses.append(current_d_loss)
            content_losses.append(current_content_loss)
            loss_ratio = current_d_loss / max(current_g_loss, 0.001)

            if batch_idx % 50 == 0:
                if not use_wgan_gp:
                    real_pred_mean = torch.sigmoid(real_output).mean().item()
                    fake_pred_mean = torch.sigmoid(fake_output).mean().item()
                else:
                    real_pred_mean = real_output.mean().item()
                    fake_pred_mean = fake_output.mean().item()

                d_trained = (
                    "YES"
                    if (
                        should_train_discriminator and current_d_loss > d_loss_threshold
                    )
                    else "NO"
                )
                print(
                    f"Batch {batch_idx}: D_loss={current_d_loss:.4f} "
                    f"(Real: {d_real_loss_item:.4f}, Fake: {d_fake_loss_item:.4f}), "
                    f"G_loss={current_g_loss:.4f}, Content={current_content_loss:.4f}, D_trained={d_trained}"
                )
                print(
                    f"  Real preds: {real_pred_mean:.4f}, Fake preds: {fake_pred_mean:.4f}, "
                    f"Loss ratio: {loss_ratio:.3f}, D_LR: {optimizer_d.param_groups[0]['lr']:.6f}"
                )

            # Clear variables and cache every few batches
            if batch_idx % 5 == 0:
                del fake_imgs, fake_imgs_gen, real_output, fake_output, fake_output_gen
                torch.cuda.empty_cache()

        # No learning rate updates needed - using fixed LR

        # Calculate average losses
        avg_g_loss = np.mean(g_losses)
        avg_d_loss = np.mean(d_losses)
        avg_content_loss = np.mean(content_losses)

        # Log to TensorBoard
        writer.add_scalar("Loss/Generator", avg_g_loss, epoch)
        writer.add_scalar("Loss/Discriminator", avg_d_loss, epoch)
        writer.add_scalar("Loss/Content", avg_content_loss, epoch)
        writer.add_scalar("LR/Generator", lr_g_fixed, epoch)
        writer.add_scalar("LR/Discriminator", lr_d_fixed, epoch)

        # Add histogram of predictions for monitoring discriminator behavior
        with torch.no_grad():
            sample_real = real_imgs[:4].clone()
            sample_age = real_age_vecs[
                :4
            ].clone()  # Use real_age_vecs (already on device)
            sample_asd = asd_vecs[:4].clone()

            # Create shifted age vectors for monitoring
            sample_shifted_age = sample_age.clone()
            new_age_vector = sample_age.clone()
            for i in range(sample_age.size(0)):
                current_age = sample_age[i].sum().item()  # Get current age
                age_shift = torch.randint(1, 4, (1,)).item()  # Random shift 1-3 years
                if current_age + age_shift > 20:
                    age_shift = 20 - current_age
                new_age = max(1, min(20, current_age + age_shift))
                sample_shifted_age[i] = torch.zeros_like(sample_age[i])
                if new_age > 0:
                    sample_shifted_age[i][: min(int(age_shift), 20)] = 1.0
                    new_age_vector[i][: min(int(new_age), 20)] = 1.0

            sample_fake = generator(sample_real, sample_shifted_age, sample_asd)
            real_preds = discriminator(sample_real, sample_age, sample_asd)
            fake_preds = discriminator(sample_fake, new_age_vector, sample_asd)

            if not use_wgan_gp:
                real_preds = torch.sigmoid(real_preds)
                fake_preds = torch.sigmoid(fake_preds)

            # Log scalar statistics instead of histograms (more reliable)
            real_mean = real_preds.mean().item()
            fake_mean = fake_preds.mean().item()
            real_std = real_preds.std().item()
            fake_std = fake_preds.std().item()

            writer.add_scalar("Predictions/Real_Mean", real_mean, epoch)
            writer.add_scalar("Predictions/Fake_Mean", fake_mean, epoch)
            writer.add_scalar("Predictions/Real_Std", real_std, epoch)
            writer.add_scalar("Predictions/Fake_Std", fake_std, epoch)

            # Additional discriminator health metrics
            pred_diff = abs(real_mean - fake_mean)
            writer.add_scalar("Discriminator/Prediction_Difference", pred_diff, epoch)

            if not use_wgan_gp:
                # For standard GAN, good discriminator should predict ~0.5 for both
                discriminator_balance = min(real_mean, 1 - fake_mean)
                writer.add_scalar(
                    "Discriminator/Balance_Score", discriminator_balance, epoch
                )

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] - G Loss: {avg_g_loss:.4f}, "
                f"D Loss: {avg_d_loss:.4f}, Content Loss: {avg_content_loss:.4f}"
            )

        # Save sample images every 5 epochs (starting from epoch 1) for better monitoring
        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Only process a small subset for visualization
            with torch.no_grad():
                sample_real = real_imgs[:2].clone()  # Only 2 samples
                sample_age = real_age_vecs[
                    :2
                ].clone()  # Use real_age_vecs (already on device)

                sample_asd = asd_vecs[:2].clone()
                save_sample_images_memory_efficient(
                    generator,
                    sample_real,
                    sample_age,
                    sample_asd,
                    epoch,
                    writer,
                    device,
                )

        # Save models every 100 epochs
        if (epoch + 1) % 100 == 0:
            result_folder = Path(
                "/projectnb/ace-genetics/jueqiw/experiment/Autism_Brain_Development/experiments/gan_weights"
            )
            result_folder.mkdir(parents=True, exist_ok=True)

            # Create unique filename with epoch number
            experiment_name = getattr(hparams, "experiment_name", "gan_training")
            model_prefix = str(result_folder / f"gan_{experiment_name}_epoch_{epoch+1}")

            save_gan_models(
                generator,
                discriminator,
                model_prefix,
                hparams=hparams,
                epoch=epoch + 1,
            )
            print(
                f"Models saved at epoch {epoch+1}: {model_prefix}_generator.pt and {model_prefix}_discriminator.pt"
            )

            # Generate and log comprehensive result plots every 100 epochs
            generate_epoch_result_plots(
                generator,
                real_imgs[:4],
                real_age_vecs[:4],
                asd_vecs[:4],
                epoch + 1,
                writer,
                device,
                hparams,
            )

        # Clear cache at the end of each epoch
        torch.cuda.empty_cache()

    writer.close()
    return generator, discriminator


def save_sample_images_memory_efficient(
    generator, real_imgs, age_vecs, asd_vecs, epoch, writer, device
):
    """Memory-efficient version of sample image saving with TensorBoard logging"""
    generator.eval()

    shifted_age_vecs = age_vecs.clone()
    new_age_vecs = age_vecs.clone()
    batch_size = real_imgs.size(0)
    for i in range(batch_size):
        current_age = age_vecs[i].sum().item()
        age_shift = torch.randint(1, 4, (1,)).item()
        if current_age + age_shift > 20:
            age_shift = 20 - current_age
        new_age = max(1, min(20, current_age + age_shift))
        shifted_age_vecs[i] = torch.zeros_like(age_vecs[i])
        new_age_vecs[i] = torch.zeros_like(age_vecs[i])
        if new_age > 0:
            shifted_age_vecs[i][: min(int(age_shift), 20)] = 1.0
            new_age_vecs[i][: min(int(new_age), 20)] = 1.0

    with torch.no_grad():
        fake_imgs = generator(real_imgs, shifted_age_vecs, asd_vecs)

    # Create comparison grid with 3 rows: real, generated, and difference
    num_samples = min(2, real_imgs.size(0))
    fig, axes = plt.subplots(3, num_samples, figsize=(8, 12))

    for i in range(num_samples):
        age = age_vecs[i].sum().item()
        new_age = new_age_vecs[i].sum().item()
        dx = "ASD" if asd_vecs[i, 0].item() > 0.5 else "Control"

        # Real images
        real_img = real_imgs[i, 0].cpu().numpy()
        axes[0, i].imshow(real_img, cmap="gray")
        axes[0, i].set_title(f"Real - Age: {age:.0f}, {dx}")
        axes[0, i].axis("off")

        # Generated images
        fake_img = fake_imgs[i, 0].cpu().numpy()
        axes[1, i].imshow(fake_img, cmap="gray")
        axes[1, i].set_title(f"Generated - Age: {new_age:.0f}, {dx}")
        axes[1, i].axis("off")

        # Difference images (Generated - Real)
        diff_img = fake_img - real_img
        im = axes[2, i].imshow(diff_img, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        axes[2, i].set_title(f"Difference (Gen - Real)\nAge: {age:.0f}→{new_age:.0f}")
        axes[2, i].axis("off")

        # Add colorbar for difference image
        plt.colorbar(im, ax=axes[2, i], shrink=0.6)

    plt.suptitle(f"GAN Results - Epoch {epoch+1}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Log to TensorBoard with descriptive title
    writer.add_figure(
        f"GAN_Images/Epoch_{epoch+1:03d}_Real_vs_Generated_vs_Diff", fig, epoch
    )
    plt.close(fig)

    # Also log individual images with more details
    for i in range(num_samples):
        age = age_vecs[i].sum().item()
        dx = "ASD" if asd_vecs[i, 0].item() > 0.5 else "Control"

        # Log real image - properly squeeze to CHW format
        real_img_tensor = (
            real_imgs[i].squeeze(0) if real_imgs[i].dim() == 4 else real_imgs[i]
        )  # Ensure CHW
        try:
            writer.add_image(
                f"Real_Images/Sample_{i+1}_Age_{age:.0f}_{dx}",
                real_img_tensor,
                epoch,
                dataformats="CHW",
            )
        except Exception as e:
            print(
                f"Error logging real image {i}: {e}, tensor shape: {real_img_tensor.shape}"
            )

        fake_img_tensor = (
            fake_imgs[i].squeeze(0) if fake_imgs[i].dim() == 4 else fake_imgs[i]
        )
        try:
            writer.add_image(
                f"Generated_Images/Sample_{i+1}_Age_{age:.0f}_{dx}",
                fake_img_tensor,
                epoch,
                dataformats="CHW",
            )
        except Exception as e:
            print(
                f"Error logging fake image {i}: {e}, tensor shape: {fake_img_tensor.shape}"
            )

    del fake_imgs
    torch.cuda.empty_cache()
    generator.train()


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
        age = row["AGE_AT_SCAN "].values[0]
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


def save_gan_models(
    generator, discriminator, filepath_prefix, hparams=None, epoch=None
):
    """Save GAN models (generator and discriminator)"""

    # Save generator
    gen_save_dict = {
        "model_state_dict": generator.state_dict(),
        "model_type": "generator",
        "epoch": epoch,
        "hparams": vars(hparams) if hasattr(hparams, "__dict__") else hparams,
        "save_timestamp": None,
    }

    # Save discriminator
    disc_save_dict = {
        "model_state_dict": discriminator.state_dict(),
        "model_type": "discriminator",
        "epoch": epoch,
        "hparams": vars(hparams) if hasattr(hparams, "__dict__") else hparams,
        "save_timestamp": None,
    }

    timestamp = datetime.datetime.now().isoformat()
    gen_save_dict["save_timestamp"] = timestamp
    disc_save_dict["save_timestamp"] = timestamp

    torch.save(gen_save_dict, f"{filepath_prefix}_generator.pt")
    torch.save(disc_save_dict, f"{filepath_prefix}_discriminator.pt")

    print(
        f"GAN models saved to {filepath_prefix}_generator.pt and {filepath_prefix}_discriminator.pt"
    )


def load_gan_models(filepath_prefix, device="cpu"):
    """Load GAN models (generator and discriminator)"""

    gen_checkpoint = torch.load(f"{filepath_prefix}_generator.pt", map_location=device)
    disc_checkpoint = torch.load(
        f"{filepath_prefix}_discriminator.pt", map_location=device
    )

    # Note: You'll need to recreate the models with the same architecture
    # This is a simplified loader - in practice you'd want to save/load architecture info
    print(
        f"GAN models loaded from {filepath_prefix}_generator.pt and {filepath_prefix}_discriminator.pt"
    )

    return gen_checkpoint, disc_checkpoint


def evaluate_gan(generator, test_loader, device):
    """Evaluate GAN generator on test set"""
    generator.eval()

    total_content_loss = 0
    num_batches = 0

    with torch.no_grad():
        for real_imgs, age_vecs, asd_vecs in test_loader:
            real_imgs = real_imgs.to(device)
            age_vecs = age_vecs.to(device)
            asd_vecs = asd_vecs.to(device)

            # Generate fake images
            fake_imgs = generator(real_imgs, age_vecs, asd_vecs)

            # Calculate content loss (L1 between real and generated)
            content_loss = F.l1_loss(fake_imgs, real_imgs)
            total_content_loss += content_loss.item()
            num_batches += 1

    avg_content_loss = total_content_loss / num_batches
    print(f"Test Content Loss (L1): {avg_content_loss:.4f}")

    return avg_content_loss


def main(hparams):
    """Main function for GAN training - Memory optimized"""
    # Set memory optimization environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cached_dataset = create_monai_dataset(cache_rate=0.5, num_workers=4)
    print(f"Loaded cached dataset with {len(cached_dataset)} samples")
    print("Splitting dataset by subjects into train/val/test (8:1:1)...")

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
    train_dataset = MRIGANDataset(cached_dataset, train_indices)
    val_dataset = MRIGANDataset(cached_dataset, val_indices)
    test_dataset = MRIGANDataset(cached_dataset, test_indices)

    # Memory optimized batch size and data loading
    batch_size = getattr(hparams, "batch_size", 16)
    print(f"Using batch size: {batch_size} for memory efficiency")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=8,
    #     pin_memory=True,
    # )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    # Create GAN models with memory-optimized parameters
    print("Creating GAN models with memory optimization...")

    # Reduced filter sizes for memory efficiency
    filters = getattr(hparams, "filters", 32)  # Reduced from 64 to 32
    latent_space = getattr(hparams, "latent_space", 64)  # Reduced from 128 to 64

    generator = create_gan_generator(
        input_shape=(1, 192, 192),
        filters=filters,
        latent_space=latent_space,
        age_dim=20,
        AD_dim=1,
        activation="tanh",
    ).to(device)

    discriminator = create_discriminator(
        input_shape=(1, 192, 192),
        filters=filters,
        ASD_dim=1,
    ).to(device)

    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(
        f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}"
    )

    epochs = getattr(hparams, "n_epochs", 100)
    lr_g = getattr(hparams, "lr_generator", 2e-4)
    lr_d = getattr(hparams, "lr_discriminator", 1e-4)  # Slower discriminator learning

    perceptual_loss_fn = PerceptualLoss(
        spatial_dims=2,
        network_type="radimagenet_resnet50",
        is_fake_3d=False,
        fake_3d_ratio=0.2,
        pretrained_path="/projectnb/ace-genetics/jueqiw/experiment/Autism_Brain_Development/pretrain_weight/RadImageNet-ResNet50_notop.pth",
    ).to(device)

    trained_generator, trained_discriminator = train_gan(
        train_loader=train_loader,
        generator=generator,
        discriminator=discriminator,
        epochs=epochs,
        lr_g=lr_g,
        lr_d=lr_d,
        perceptual_loss_fn=perceptual_loss_fn,
        hparams=hparams,
    )

    print("Training completed! Evaluating on test set...")

    # Evaluate on test set
    test_content_loss = evaluate_gan(trained_generator, test_loader, device)
    print(f"Final test content loss: {test_content_loss:.4f}")

    # Save trained models
    result_folder = Path(
        "/projectnb/ace-genetics/jueqiw/experiment/Autism_Brain_Development/experiments/gan_weights"
    )
    os.makedirs(str(result_folder), exist_ok=True)
    model_prefix = str(result_folder / f"gan_{hparams.experiment_name}")
    save_gan_models(
        trained_generator,
        trained_discriminator,
        model_prefix,
        hparams=hparams,
        epoch=epochs,
    )

    print(
        f"GAN models saved to {model_prefix}_generator.pt and {model_prefix}_discriminator.pt"
    )
    print("\n" + "=" * 50)
    print("TESTING MODEL LOADING...")
    print("=" * 50)
    try:
        gen_checkpoint = torch.load(f"{model_prefix}_generator.pt", map_location=device)
        disc_checkpoint = torch.load(
            f"{model_prefix}_discriminator.pt", map_location=device
        )

        # Create fresh model instances
        loaded_generator = create_gan_generator(
            input_shape=(1, 192, 192),
            filters=filters,
            latent_space=latent_space,
            age_dim=20,
            AD_dim=1,
            activation="tanh",
        ).to(device)

        loaded_discriminator = create_discriminator(
            input_shape=(1, 192, 192),
            filters=filters,
            age_dim=25,
            ASD_dim=1,
        ).to(device)

        loaded_generator.load_state_dict(gen_checkpoint["model_state_dict"])
        loaded_discriminator.load_state_dict(disc_checkpoint["model_state_dict"])

        print(f"Generator trained for: {gen_checkpoint.get('epoch', 'unknown')} epochs")
        print(
            f"Discriminator trained for: {disc_checkpoint.get('epoch', 'unknown')} epochs"
        )

        loaded_generator.eval()
        loaded_discriminator.eval()

        with torch.no_grad():
            test_batch = next(iter(test_loader))
            test_imgs, test_age_vecs, test_asd_vecs = test_batch
            test_imgs = test_imgs[:2].to(device)
            test_age_vecs = test_age_vecs[:2].to(device)
            test_asd_vecs = test_asd_vecs[:2].to(device)
            loaded_fake = loaded_generator(test_imgs, test_age_vecs, test_asd_vecs)
            loaded_d_real = loaded_discriminator(
                test_imgs, test_age_vecs, test_asd_vecs
            )
            loaded_d_fake = loaded_discriminator(
                loaded_fake, test_age_vecs, test_asd_vecs
            )

            original_fake = trained_generator(test_imgs, test_age_vecs, test_asd_vecs)

        print("Model loading test completed successfully!")

    except Exception as e:
        print(f"Error during model loading test: {e}")
        import traceback

        traceback.print_exc()

    print("=" * 50)

    # Generate sample images with the original trained model
    generate_sample_images(trained_generator, test_loader, hparams, device)

    return trained_generator, trained_discriminator


def generate_sample_images(generator, test_loader, hparams, device, num_samples=8):
    """Generate sample images across different ages and conditions with TensorBoard logging"""
    generator.eval()

    # Create output directory
    output_dir = Path("gan_generated_samples")
    os.makedirs(str(output_dir), exist_ok=True)

    # Set up TensorBoard writer for final results
    log_dir = "/projectnb/ace-genetics/jueqiw/experiment/Autism_Brain_Development/experiments/tensorboard"
    if hparams and hasattr(hparams, "experiment_name"):
        log_dir = os.path.join(log_dir, f"{hparams.experiment_name}_final_results")
    writer = SummaryWriter(log_dir=log_dir)

    with torch.no_grad():
        # Get a batch from test loader
        test_batch = next(iter(test_loader))
        real_imgs, age_vecs, asd_vecs = test_batch

        real_imgs = real_imgs[:num_samples].to(device)
        age_vecs = age_vecs[:num_samples].to(device)
        asd_vecs = asd_vecs[:num_samples].to(device)

        # Create shifted age vectors for age manipulation
        shifted_age_vecs = age_vecs.clone()
        new_age_vecs = age_vecs.clone()
        for i in range(num_samples):
            current_age = age_vecs[i].sum().item()
            age_shift = torch.randint(1, 4, (1,)).item()  # Random shift 1-3 years
            if current_age + age_shift > 20:
                age_shift = 20 - current_age
            new_age = max(1, min(20, current_age + age_shift))
            shifted_age_vecs[i] = torch.zeros_like(age_vecs[i])
            new_age_vecs[i] = torch.zeros_like(age_vecs[i])
            if new_age > 0:
                shifted_age_vecs[i][: min(int(age_shift), 20)] = 1.0
                new_age_vecs[i][: min(int(new_age), 20)] = 1.0

        fake_imgs = generator(real_imgs, shifted_age_vecs, asd_vecs)
        fig, axes = plt.subplots(3, num_samples, figsize=(2 * num_samples, 6))

        for i in range(num_samples):
            real_img = real_imgs[i, 0].cpu().numpy()
            age = age_vecs[i].sum().item()
            new_age = new_age_vecs[i].sum().item()
            dx = "ASD" if asd_vecs[i, 0].item() > 0.5 else "Control"

            # Real images
            axes[0, i].imshow(real_img, cmap="gray")
            axes[0, i].set_title(f"Real\nAge:{age:.0f}, {dx}")
            axes[0, i].axis("off")

            # Generated images
            fake_img = fake_imgs[i, 0].cpu().numpy()
            axes[1, i].imshow(fake_img, cmap="gray")
            axes[1, i].set_title(f"Generated\nAge:{new_age:.0f}, {dx}")
            axes[1, i].axis("off")

            # Difference images
            diff_img = fake_img - real_img
            im = axes[2, i].imshow(diff_img, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
            axes[2, i].set_title(f"Difference\n{age:.0f}→{new_age:.0f}")
            axes[2, i].axis("off")

            # Add colorbar for difference image
            plt.colorbar(im, ax=axes[2, i], shrink=0.6)

        plt.suptitle(
            "Final GAN Results - Test Set Samples", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(
            str(output_dir / "sample_comparison.png"), dpi=150, bbox_inches="tight"
        )

        # Log the comparison figure to TensorBoard
        writer.add_figure("Final_Results/Test_Set_Comparison", fig, 0)
        plt.close()

        # Log individual test images to TensorBoard with detailed titles
        for i in range(num_samples):
            age = age_vecs[i].sum().item()
            dx = "ASD" if asd_vecs[i, 0].item() > 0.5 else "Control"

            # Log real test image - properly squeeze to CHW format
            real_img_tensor = (
                real_imgs[i].squeeze(0) if real_imgs[i].dim() == 4 else real_imgs[i]
            )
            try:
                writer.add_image(
                    f"Test_Real/Sample_{i+1:02d}_Age_{age:.0f}_{dx}",
                    real_img_tensor,
                    0,
                    dataformats="CHW",
                )
            except Exception as e:
                print(
                    f"Error logging test real image {i}: {e}, tensor shape: {real_img_tensor.shape}"
                )

            # Log generated test image - properly squeeze to CHW format
            fake_img_tensor = (
                fake_imgs[i].squeeze(0) if fake_imgs[i].dim() == 4 else fake_imgs[i]
            )
            try:
                writer.add_image(
                    f"Test_Generated/Sample_{i+1:02d}_Age_{age:.0f}_{dx}",
                    fake_img_tensor,
                    0,
                    dataformats="CHW",
                )
            except Exception as e:
                print(
                    f"Error logging test fake image {i}: {e}, tensor shape: {fake_img_tensor.shape}"
                )

        print(f"Sample images saved to {output_dir}/sample_comparison.png")
        print(f"TensorBoard logs saved to {log_dir}")

    writer.close()
    generator.train()


if __name__ == "__main__":
    parser = ArgumentParser(description="GAN Training for MRI Data", add_help=False)
    add_argument(parser)

    args = parser.parse_args()
    print("Starting GAN training with conditional age and ASD information...")
    main(args)
