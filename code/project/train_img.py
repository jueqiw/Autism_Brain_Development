from utils.utils import get_ABIDE_II_subject_followup
from utils.transforms import get_train_data_transform

import monai
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch

import nibabel as nib
import ants
import numpy as np
import matplotlib.pyplot as plt


def visualize(followup_img, baseline_img, followup_mask, baseline_mask):
    followup_img = followup_img * followup_mask
    baseline_img = baseline_img * baseline_mask

    # Convert torch tensors to numpy arrays if needed
    if hasattr(followup_img, "numpy"):
        followup_img = followup_img.numpy()
    if hasattr(baseline_img, "numpy"):
        baseline_img = baseline_img.numpy()
    if hasattr(followup_mask, "numpy"):
        followup_mask = followup_mask.numpy()
    if hasattr(baseline_mask, "numpy"):
        baseline_mask = baseline_mask.numpy()

    # Ensure arrays are float32 and remove channel dimension if present
    followup_img = np.squeeze(followup_img).astype(np.float32)
    baseline_img = np.squeeze(baseline_img).astype(np.float32)
    followup_mask = np.squeeze(followup_mask).astype(np.float32)
    baseline_mask = np.squeeze(baseline_mask).astype(np.float32)

    # Debug information
    print(f"Baseline shape: {baseline_img.shape}, dtype: {baseline_img.dtype}")
    print(f"Followup shape: {followup_img.shape}, dtype: {followup_img.dtype}")
    print(f"Baseline range: [{baseline_img.min():.4f}, {baseline_img.max():.4f}]")
    print(f"Followup range: [{followup_img.min():.4f}, {followup_img.max():.4f}]")

    # Normalize images to [0, 1] range if they're not already
    if baseline_img.max() > 1.0:
        baseline_img = baseline_img / baseline_img.max()
    if followup_img.max() > 1.0:
        followup_img = followup_img / followup_img.max()

    try:
        baseline_img = baseline_img * baseline_mask
        followup_img = followup_img * followup_mask
        # Convert numpy arrays to ANTs images
        fixed_img = ants.from_numpy(baseline_img)
        moving_img = ants.from_numpy(followup_img)

        # Set spacing and origin (important for registration)
        fixed_img.set_spacing([1.0, 1.0, 1.0])
        moving_img.set_spacing([1.0, 1.0, 1.0])
        fixed_img.set_origin([0.0, 0.0, 0.0])
        moving_img.set_origin([0.0, 0.0, 0.0])

        # Perform registration with more robust parameters
        reg = ants.registration(
            fixed=fixed_img,
            moving=moving_img,
            type_of_transform="SyN",
        )

        # Create jacobian determinant
        jacobian = ants.create_jacobian_determinant_image(
            domain_image=fixed_img, tx=reg["fwdtransforms"][0], do_log=True
        )
        jac_data = jacobian.numpy()
        return jac_data

    except Exception as e:
        print(f"Registration failed with error: {e}")
        print("Returning zero array as fallback")
        return np.zeros_like(baseline_img)


if __name__ == "__main__":
    samples = get_ABIDE_II_subject_followup()
    train_transform = get_train_data_transform()
    # val_transform = get_val_data_augmentation_transform()
    train_ds = Dataset(data=samples, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=8)
    jacobian_maps = []

    for step, train_batch_data in enumerate(train_loader):
        # Get the subject ID from the batch data
        baseline_img = train_batch_data["baseline_img"][0]
        followup_img = train_batch_data["followup_img"][0]
        baseline_mask = train_batch_data["baseline_mask"][0]
        followup_mask = train_batch_data["followup_mask"][0]
        # Visualize the subject's MRI data
        jacobian_map = visualize(
            followup_img, baseline_img, followup_mask, baseline_mask
        )
        jacobian_maps.append(jacobian_map)

        # Get ages for the subject
        baseline_age, followup_age = (
            train_batch_data["baseline_age"][0],
            train_batch_data["followup_age"][0],
        )

    mean_jacobian_map = np.mean(jacobian_maps, axis=0)
    # take one slice from the 3D image for visualization
    mean_jacobian_map_2D = mean_jacobian_map[:, :, mean_jacobian_map.shape[2] // 2]
    plt.imshow(mean_jacobian_map_2D, cmap="gray")
    plt.title("Mean Jacobian Map")
    plt.colorbar()
    # save image
    plt.savefig("/project/ace-genetics/jueqiw/mean_jacobian_map.png")
    # close the plot
    plt.close()
