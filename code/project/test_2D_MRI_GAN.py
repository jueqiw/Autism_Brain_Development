import os
import re
from glob import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from monai.transforms import (
    Compose,
    LoadImaged,
    Resized,
    ScaleIntensityRanged,
    ToTensord,
    MapTransform,
)
from monai.data import NibabelReader
from pathlib import Path
import pandas as pd
import ants
import sys
from skimage.transform import resize

from models.gan_generator import create_gan_generator


baseline_dir = "/projectnb/ace-ig/ABIDE/ABIDE_II_Baseline_2D_npy/coronal"
followup_dir = "/projectnb/ace-ig/ABIDE/ABIDE_II_Followup_2D_npy/coronal"
output_dir = "/projectnb/ace-genetics/grace/GAN_output/gan_gan_d_1_lr_1e-4_gr_2e-4_per_0.1_content_1_adver_10.0_patch_GAN_without_scheduler_epoch_500_generator"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LoadNumpyd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            npy_path = d[key]
            img = np.load(npy_path)
            # Ensure channel dimension exists
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)  # (1, H, W)
            d[key] = img.astype(np.float32)
        return d


# --- Preprocessing transform same as training ---
preproc = Compose(
    [
        LoadNumpyd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(192, 192)),
        ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=1.0, b_min=-1.0, b_max=1.0, clip=True
        ),
        ToTensord(keys=["image"]),
    ]
)


def create_age_vector(age_float):
    age_int = max(0, min(20, int(round(age_float))))
    age_vector = torch.zeros(20, dtype=torch.float32)
    if age_int > 0:
        age_vector[:age_int] = 1.0
    return age_vector


def create_asd_vector(dx_group):
    # Returns: condition: 1D numpy array with diagnosis value (1 channel)
    # Use single value instead of one-hot: 0.0 for ASD, 1.0 for Control
    return torch.tensor([float(dx_group)], dtype=torch.float32)


def get_ages_labels(
    sub_id,
):  # all subjects in ABIDE II Longitudinal dataset are under 21
    sub_id_int = int(sub_id)
    csv_path = "/projectnb/ace-ig/ABIDE/ABIDEII_Long_Composite_Phenotypic.csv"
    df = pd.read_csv(csv_path)
    if df[(df["SUB_ID"] == sub_id_int)].empty:
        print(f"Subject {sub_id} not found in phenotypic CSV.")
        return None, None, None
    try:
        dx_group = (
            int(
                df[(df["SUB_ID"] == sub_id_int) & (df["SESSION"] == "Baseline")][
                    "DX_GROUP"
                ].values[0]
            )
            - 1
        )
        baseline_age = float(
            df[(df["SUB_ID"] == sub_id_int) & (df["SESSION"] == "Baseline")][
                "AGE_AT_SCAN "
            ].values[0]
        )
        followup_age = float(
            df[(df["SUB_ID"] == sub_id_int) & (df["SESSION"] == "Followup_1")][
                "AGE_AT_SCAN "
            ].values[0]
        )
        return baseline_age, followup_age, dx_group
    except Exception as e:
        print(f"Error reading ages/labels for subject {sub_id}: {e}")
        return None, None, None


# --- Load GAN generator model ---
filters = 32
latent_space = 64

generator = create_gan_generator(
    input_shape=(1, 192, 192),
    filters=filters,
    latent_space=latent_space,
    age_dim=20,
    AD_dim=1,
    activation="tanh",
).to(device)

weights = "/projectnb/ace-genetics/jueqiw/experiment/Autism_Brain_Development/experiments/gan_weights/gan_gan_d_1_lr_1e-4_gr_2e-4_per_0.1_content_1_adver_10.0_patch_GAN_without_scheduler_epoch_500_generator.pt"
checkpoint = torch.load(weights, map_location=device)
generator.load_state_dict(checkpoint["model_state_dict"])
generator.eval()

# --- Main testing loop ---
baseline_imgs = glob(os.path.join(baseline_dir, "*.npy"))
print(f"Found {len(baseline_imgs)} baseline images for testing.")


def get_roi_avg_jacobian(jac_data_2d, roi_mask_2d):
    """
    Visualize average Jacobian determinant values for each ROI in a 2D slice.
    Parameters:
    -----------
    jac_data_2d : np.ndarray
        2D numpy array of Jacobian determinant values for one slice.
    roi_mask_2d : np.ndarray
        2D numpy array with ROI labels per voxel (0 means background).
    Returns:
    --------
    avg_jac_map : np.ndarray
        2D array where each ROI voxel has the average jacobian value of that ROI.
    """

    # resize ROI mask to match jacobian data shape
    roi_resized = resize(
        roi_mask_2d,
        jac_data_2d.shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(roi_mask_2d.dtype)

    # Get unique ROI labels ignoring background (0)
    roi_labels = np.unique(roi_resized)
    roi_labels = roi_labels[roi_labels != 0]

    avg_jac_map = np.zeros_like(roi_resized, dtype=np.float32)

    for label in roi_labels:
        mask = roi_resized == label
        if np.any(mask):
            avg_val = jac_data_2d[mask].mean()
            avg_jac_map[mask] = avg_val  # average jacobian value for this ROI

    return avg_jac_map


def apply_mask(img, mask):
    if torch.is_tensor(mask):  # turn original mask from tensor to numpy
        mask_np = mask.cpu().numpy().astype(bool)
    else:
        mask_np = mask.astype(bool)
    masked_img = img.copy()

    print("masked_img shape:", masked_img.shape)
    print("mask_np shape:", mask_np.shape)

    masked_img[~mask_np] = 1  # background areas = 1 for the jacobians
    return masked_img


for baseline_path in baseline_imgs:
    basename = os.path.basename(baseline_path)
    # Extract subject ID using filename pattern: "['sub_id']_0.npy"
    match = re.search(r"\['(\d+)'\]_0\.npy", basename)
    if not match:
        continue

    sub_id = match.group(1)
    followup_path = os.path.join(followup_dir, f"['{sub_id}']_0.npy")
    if not os.path.exists(followup_path):
        print(f"Follow-up image for subject {sub_id} not found, skipping.")
        continue

    baseline_age, followup_age, dx_group = get_ages_labels(sub_id)
    if baseline_age is None or followup_age is None or dx_group is None:
        continue

    try:
        # Preprocess baseline image
        preproc_out = preproc({"image": baseline_path})
        baseline_img = preproc_out["image"]
        if baseline_img.shape[0] > 1:
            baseline_img = baseline_img.mean(dim=0, keepdim=True)
        baseline_img = baseline_img.unsqueeze(0).to(device)  # (1, 1, H, W)

        brain_mask = baseline_img > 0.001  # Adjust threshold as appropriate
        brain_mask = brain_mask[0, 0, :, :].cpu().numpy().astype(bool)

        # Preprocess ground truth follow-up image
        preproc_out_fu = preproc({"image": followup_path})
        followup_img_tensor = preproc_out_fu["image"]
        if followup_img_tensor.shape[0] > 1:
            followup_img_tensor = followup_img_tensor.mean(dim=0, keepdim=True)
        followup_img = followup_img_tensor.cpu().numpy()[0]

        # Create diagnosis vector (0 for ASD, 1 for Control)
        asd_vec = create_asd_vector(dx_group).unsqueeze(0).to(device)  # (1, 1)

        age_gap = round(followup_age - baseline_age)
        age_vec = create_age_vector(age_gap).unsqueeze(0).to(device)  # (1, 20)
        recon = (
            generator(baseline_img, age_vec, asd_vec).detach().cpu().squeeze().numpy()
        )

        # Process baseline and ground truth followup for ants registration as before
        baseline_img_np = baseline_img[0, 0].cpu().numpy()
        baseline_ants = ants.from_numpy(baseline_img_np)
        baseline_ants = ants.iMath(baseline_ants, "Normalize")

        followup_ants = ants.from_numpy(followup_img)
        followup_ants = ants.iMath(followup_ants, "Normalize")

        # Register baseline -> gt followup jacobian
        gt_reg = ants.registration(
            fixed=baseline_ants, moving=followup_ants, type_of_transform="SyN"
        )
        gt_jacobian = ants.create_jacobian_determinant_image(
            domain_image=baseline_ants, tx=gt_reg["fwdtransforms"][0], do_log=False
        )
        gt_jac_data = gt_jacobian.numpy()

        recon_ants = ants.from_numpy(recon)
        recon_ants = ants.iMath(recon_ants, "Normalize")
        recon_reg = ants.registration(
            fixed=baseline_ants, moving=recon_ants, type_of_transform="SyN"
        )
        recon_jacobian = ants.create_jacobian_determinant_image(
            domain_image=baseline_ants, tx=recon_reg["fwdtransforms"][0], do_log=False
        )
        recon_jac_data = recon_jacobian.numpy()

        gt_jac_data_masked = apply_mask(gt_jac_data, brain_mask)
        recon_jac_data_masked = apply_mask(recon_jac_data, brain_mask)

        norm = TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=1.5)
        fig, axs = plt.subplots(2, 6, figsize=(30, 13), constrained_layout=True)

        # ======= First row =======
        # 1) GT Baseline image
        axs[0, 0].imshow(baseline_img_np, cmap="gray")
        axs[0, 0].set_title(f"GT Baseline\nAge: {baseline_age:.1f}", fontsize=35)
        axs[0, 0].axis("off")
        axs[0, 0].set_aspect("equal")

        # 2) GT Follow-up image
        axs[0, 1].imshow(followup_img, cmap="gray")
        axs[0, 1].set_title(f"GT Follow-up\nAge: {followup_age:.1f}", fontsize=35)
        axs[0, 1].axis("off")
        axs[0, 1].set_aspect("equal")

        # 3) Recon Follow-up image
        axs[0, 2].imshow(recon, cmap="gray")
        axs[0, 2].set_title(f"Recon Follow-up", fontsize=35)
        axs[0, 2].axis("off")
        axs[0, 2].set_aspect("equal")

        # 4) Jacobian: GT Baseline → GT Follow-Up
        im_gt = axs[0, 3].imshow(gt_jac_data_masked, cmap="coolwarm", norm=norm)
        axs[0, 3].set_title(f"Jacobian:\nGT Baseline → GT Follow-up", fontsize=30)
        axs[0, 3].axis("off")
        axs[0, 3].set_aspect("equal")
        gt_cbar = fig.colorbar(im_gt, ax=axs[0, 3], fraction=0.046, pad=0.02)
        gt_cbar.ax.tick_params(labelsize=25)

        # 5) Jacobian: GT Baseline → Recon Follow-Up
        im_floor = axs[0, 4].imshow(recon_jac_data_masked, cmap="coolwarm", norm=norm)
        axs[0, 4].set_title(f"Jacobian:\nGT Baseline → Recon", fontsize=30)
        axs[0, 4].axis("off")
        axs[0, 4].set_aspect("equal")
        recon_cbar = fig.colorbar(im_floor, ax=axs[0, 4], fraction=0.046, pad=0.02)
        recon_cbar.ax.tick_params(labelsize=25)

        # 6) Jacobian difference (GT - Recon)
        jac_diff = abs(gt_jac_data_masked - recon_jac_data_masked)
        # jac_diff_masked = apply_mask(jac_diff, brain_mask)
        im_diff = axs[0, 5].imshow(jac_diff, cmap="winter")
        axs[0, 5].set_title(f"Jacobian Diff:\n|GT - Recon|", fontsize=30)
        axs[0, 5].axis("off")
        axs[0, 5].set_aspect("equal")
        diff_cbar = fig.colorbar(im_diff, ax=axs[0, 5], fraction=0.046, pad=0.02)
        diff_cbar.ax.tick_params(labelsize=25)

        # ======= Second row (3 ROI Avg Jacobian maps, rest empty) =======
        roi_mask = np.load(
            "/projectnb/ace-genetics/grace/Autism_Brain_Development/code/project/transformed_roi.npy"
        )
        slice_idx = roi_mask.shape[2] // 2
        roi_slice = roi_mask[:, :, slice_idx]
        roi_slice = np.rot90(roi_slice, k=1)

        gt_avg_jac_map = get_roi_avg_jacobian(gt_jac_data, roi_slice)
        recon_avg_jac_map = get_roi_avg_jacobian(recon_jac_data, roi_slice)
        diff_avg_jac_map = abs(gt_avg_jac_map - recon_avg_jac_map)

        # 1) GT Baseline -> GT Follow-up ROI avg jacobian
        im_roi_gt = axs[1, 0].imshow(gt_avg_jac_map, cmap="coolwarm", norm=norm)
        axs[1, 0].set_title("ROI Avg Jacobian:\nGT Baseline → Follow-up", fontsize=25)
        axs[1, 0].axis("off")
        axs[1, 0].set_aspect("equal")
        cbar_roi_gt = fig.colorbar(im_roi_gt, ax=axs[1, 0], fraction=0.046, pad=0.02)
        cbar_roi_gt.ax.tick_params(labelsize=20)

        # 2) GT Baseline -> Recon Follow-up ROI avg jacobian
        im_roi_recon = axs[1, 1].imshow(recon_avg_jac_map, cmap="coolwarm", norm=norm)
        axs[1, 1].set_title("ROI Avg Jacobian:\nGT Baseline → Recon", fontsize=25)
        axs[1, 1].axis("off")
        axs[1, 1].set_aspect("equal")
        cbar_roi_recon = fig.colorbar(
            im_roi_recon, ax=axs[1, 1], fraction=0.046, pad=0.02
        )
        cbar_roi_recon.ax.tick_params(labelsize=20)

        # 3) ROI Avg Jacobian Diff
        im_roi_diff = axs[1, 2].imshow(diff_avg_jac_map, cmap="winter")
        axs[1, 2].set_title("ROI Avg Jacobian Diff:\n|GT - Recon|", fontsize=25)
        axs[1, 2].axis("off")
        axs[1, 2].set_aspect("equal")
        cbar_roi_diff = fig.colorbar(
            im_roi_diff, ax=axs[1, 2], fraction=0.046, pad=0.02
        )
        cbar_roi_diff.ax.tick_params(labelsize=20)

        # Empty plots in the remaining 3 columns of second row
        for empty_col in range(3, 6):
            axs[1, empty_col].axis("off")

        plt.suptitle(
            f"Subject {sub_id} - Diagnosis: {'ASD' if dx_group == 0 else 'Control'}",
            fontsize=45,
        )

        save_path = os.path.join(output_dir, f"{sub_id}_multi_gap_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved visualization for subject {sub_id} at {save_path}")

    except Exception as e:
        print(f"Error processing subject {sub_id}: {e}")
        raise  # get full traceback
