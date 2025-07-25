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
    # map the value of jacobian to [-1, 1] from the range [0.85, 1.15]
    jacobian_norm = jacobian - 1.0  # scale to [-0.15, 0.15]
    scale = 1.0 / 0.15  # scale to [-1, 1]
    jacobian_norm = jacobian_norm * scale  # now in [-1, 1]
    jacobian_norm = np.clip(jacobian_norm, -1.0, 1.0)  # ensure within bounds
    return jacobian_norm


def postprocess_jacobian(jacobian_norm):
    return jacobian_norm * 0.15 + 1.0  # scale back to [0.85, 1.15]


# ------------------ Dataset ------------------
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
        jacobian = torch.from_numpy(jacobian).float()  # Remove extra unsqueeze(0)

        return input_pair, jacobian


def focal_l1_loss(gt, pred, alpha=0.25, gamma=2.0, eps=1e-8, scale_factor=10.0):
    """
    Focal Loss variant for L1 - focuses on hard examples
    Automatically handles class imbalance without manual weighting
    Added scale_factor to make loss magnitude more reasonable
    """
    l1_loss = torch.abs(pred - gt)

    # Convert L1 loss to "confidence" score (lower loss = higher confidence)
    pt = torch.exp(-l1_loss)  # High pt for easy examples, low pt for hard examples

    # Focal weight: down-weight easy examples, up-weight hard examples
    focal_weight = alpha * (1 - pt) ** gamma

    focal_l1 = focal_weight * l1_loss * scale_factor
    return focal_l1.mean()


def smooth_focal_l1_loss(gt, pred, alpha=1.0, gamma=1.0, eps=1e-8, scale_factor=1.0):
    """
    Smoother version of focal loss - less aggressive, more stable
    """
    l1_loss = torch.abs(pred - gt)

    # More conservative focal weighting
    # Use sigmoid instead of exp for smoother gradients
    pt = torch.sigmoid(-l1_loss * 5.0)  # Scaled sigmoid for smoother curve

    # Less aggressive focal weight
    focal_weight = alpha * (1 - pt) ** gamma + eps  # Add epsilon for stability

    focal_l1 = focal_weight * l1_loss * scale_factor
    return focal_l1.mean()


def adaptive_balanced_l1_loss(
    gt,
    pred,
    pos_weight=None,
    weight_clamp_max=20.0,
    eps=1e-8,
    brain_mask_threshold=0.01,
    min_roi_weight=2.0,
):
    """
    More conservative balanced L1 loss that prevents model collapse
    """
    # Create brain mask
    brain_mask = (torch.abs(gt) > brain_mask_threshold).float()

    # Calculate statistics
    brain_pixels = brain_mask.sum()
    roi_pixels = ((gt != 0).float() * brain_mask).sum()
    normal_pixels = brain_pixels - roi_pixels

    # More conservative weight calculation
    if pos_weight is None and roi_pixels > eps and normal_pixels > eps:
        # Use square root to reduce extreme ratios
        raw_ratio = normal_pixels / (roi_pixels + eps)
        pos_weight = torch.sqrt(raw_ratio).clamp(min_roi_weight, weight_clamp_max)
        pos_weight = pos_weight.item()
    elif pos_weight is None:
        pos_weight = min_roi_weight

    deformation_mask = (gt != 0).float()

    # More balanced weight scheme
    outside_brain_penalty = weight_clamp_max  # Same as max ROI weight
    weights = torch.where(
        brain_mask == 0,
        outside_brain_penalty,
        torch.where(deformation_mask == 1, pos_weight, 1.0),
    )

    l1 = torch.abs(pred - gt)
    weighted_l1 = weights * l1

    return weighted_l1.mean()


def balanced_weighted_l1_loss(
    gt,
    pred,
    pos_weight=None,
    weight_clamp_max=100.0,
    eps=1e-8,
    brain_mask_threshold=0.01,
):
    """
    Balanced L1 loss with automatic positive class weighting
    Includes brain masking to prevent predictions outside brain
    """
    # Create brain mask - areas where there's any signal (brain tissue)
    # Assuming background areas have very small or zero values in both original and transformed images
    brain_mask = (torch.abs(gt) > brain_mask_threshold).float()

    # Calculate positive class ratio for automatic weighting (only within brain)
    if pos_weight is None:
        brain_pixels = brain_mask.sum()
        positive_pixels = (
            (gt != 0).float() * brain_mask
        ).sum()  # Non-zero within brain
        negative_pixels = brain_pixels - positive_pixels  # Zero within brain

        if positive_pixels > eps and negative_pixels > eps:
            pos_weight = negative_pixels / (positive_pixels + eps)
            pos_weight = min(pos_weight, weight_clamp_max)
        else:
            pos_weight = 1.0

    # Create deformation mask (areas with significant Jacobian change)
    deformation_mask = (gt != 0).float()

    # Weight scheme:
    # - Outside brain: very high penalty (discourage predictions)
    # - Inside brain, normal tissue: 1x weight
    # - Inside brain, deformed tissue: pos_weight
    outside_brain_penalty = weight_clamp_max * 2  # Heavy penalty for outside brain
    weights = torch.where(
        brain_mask == 0,
        outside_brain_penalty,  # Outside brain
        torch.where(deformation_mask == 1, pos_weight, 1.0),
    )  # Inside brain

    l1 = torch.abs(pred - gt)
    weighted_l1 = weights * l1

    return weighted_l1.mean()


def get_loss_function(loss_type, hparams):
    """
    Factory function to get the appropriate loss function based on hyperparameters
    """
    if loss_type == "focal_l1":

        def loss_fn(gt, pred):
            return focal_l1_loss(
                gt,
                pred,
                alpha=hparams.focal_alpha,
                gamma=hparams.focal_gamma,
                scale_factor=hparams.focal_scale_factor,
            )

        return loss_fn

    elif loss_type == "smooth_focal_l1":

        def loss_fn(gt, pred):
            return smooth_focal_l1_loss(
                gt,
                pred,
                alpha=hparams.focal_alpha,
                gamma=hparams.focal_gamma,
                scale_factor=hparams.focal_scale_factor,
            )

        return loss_fn

    elif loss_type == "balanced_weighted_l1":

        def loss_fn(gt, pred):
            return balanced_weighted_l1_loss(
                gt,
                pred,
                pos_weight=hparams.pos_weight,
                weight_clamp_max=hparams.weight_clamp_max,
                brain_mask_threshold=hparams.brain_mask_threshold,
            )

        return loss_fn

    elif loss_type == "adaptive_balanced_l1":

        def loss_fn(gt, pred):
            return adaptive_balanced_l1_loss(
                gt,
                pred,
                pos_weight=hparams.pos_weight,
                weight_clamp_max=hparams.weight_clamp_max,
                brain_mask_threshold=hparams.brain_mask_threshold,
            )

        return loss_fn

    elif loss_type == "simple_l1":

        def loss_fn(gt, pred):
            return F.l1_loss(gt, pred)

        return loss_fn

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def dice_l1_loss(gt, pred, smooth=1e-6):
    """
    Combination of Dice loss (for overlap) and L1 loss (for regression)
    Good for small object detection in regression tasks
    """
    # L1 component
    l1_loss = F.l1_loss(pred, gt)

    # Dice component for non-zero regions
    gt_mask = (gt != 0).float()
    pred_mask = (
        torch.abs(pred) > 0.1
    ).float()  # Threshold for "significant" prediction

    intersection = (gt_mask * pred_mask).sum()
    union = gt_mask.sum() + pred_mask.sum()

    dice_score = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice_score

    # Combine losses
    return l1_loss + dice_loss


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


  def l1_loss(pred, target):
      l1_loss = torch.abs(pred - target)
      return l1_loss.mean()


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

            # Fix shape mismatch - squeeze extra dimensions
            if output.dim() > y.dim():
                output = output.squeeze()

            loss = l1_loss(y, output)
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

                # Fix shape mismatch - squeeze extra dimensions
                if output_val.dim() > y_val.dim():
                    output_val = output_val.squeeze()

                val_loss = l1_loss(y_val, output_val)
                total_val_loss += val_loss.item()

                # Only store one batch for logging and move to CPU to save memory
                if batch_idx == 0 and (epoch + 1) % 10 == 0:  # Only when needed
                    # Take first 4 samples and move to CPU immediately
                    sample_inputs = x_val[:4].detach().cpu()
                    sample_predictions = output_val[:4].detach().cpu()
                    sample_targets = y_val[:4].detach().cpu()

        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        # Log validation loss to TensorBoard
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

        # Update learning rate scheduler
        if hasattr(hparams, "scheduler_type") and hparams.scheduler_type == "plateau":
            scheduler.step(avg_val_loss)  # ReduceLROnPlateau needs the metric
        else:
            scheduler.step()  # Other schedulers don't need metric

        # Log current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Learning_Rate", current_lr, epoch)

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
                # Move back to GPU temporarily for logging
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

            print(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}"
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


def main(hparams):
    original, transformed, jacobians = load_data()

    model = UNet(
        spatial_dims=2,
        in_channels=2,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

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

    torch.save(model.state_dict(), f"all_model_monai_{hparams.experiment_name}.pt")


if __name__ == "__main__":
    parser = ArgumentParser(description="Trainer args", add_help=False)
    add_argument(parser)
    main(parser.parse_args())
