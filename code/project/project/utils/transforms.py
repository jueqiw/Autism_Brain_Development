from typing import Dict, Tuple

import torch
import numpy as np
from monai.data import NibabelReader
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    Cropd,
    CenterSpatialCropd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityd,
    SpatialResampled,
    SpatialCropd,
    SpatialPadd,
    MaskIntensityd,
    RandSpatialCropd,
    Spacingd,
    ToTensord,
    MapTransform,
    RandRotated,
    RandZoomd,
    RandAffined,
)


def sample_directions_training(
    adjacent_bvec: np.ndarray, center_bvec: np.ndarray, n_neighbor_directions: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    n_center_directions = center_bvec.shape[1]  # shape 3, 60
    num_centers = 1

    center_indices = np.random.choice(
        n_center_directions, size=num_centers, replace=False
    )

    center_vec = center_bvec[:, center_indices[0]]
    similarity = np.dot(adjacent_bvec.T, center_vec)

    sorted_indices = np.argsort(-similarity)
    adjacent_indices = sorted_indices[:n_neighbor_directions]

    return center_indices[0], adjacent_indices


def sample_directions_testing(
    adjacent_bvec: np.ndarray,
    center_bvec: np.ndarray,
    n_direction: int,
    n_neighbor_directions: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    center_vec = center_bvec[:, n_direction]
    similarity = np.dot(adjacent_bvec.T, center_vec)

    sorted_indices = np.argsort(-similarity)
    adjacent_indices = sorted_indices[:n_neighbor_directions]

    return n_direction, adjacent_indices


class GetCenterAdjacentImgTest(MapTransform):
    def __init__(self, keys):
        MapTransform.__init__(self, keys)

    def __call__(self, x: Dict):
        center_indices, adjacent_indices = sample_directions_testing(
            x["adjacent_direction"],
            x["center_direction"],
            int(x["n_direction"]),
            n_neighbor_directions=int(x["n_neighbor_directions"]),
        )
        center_image = x["center_img"][center_indices, :, :]
        adjacent_images = x["adjacent_img"][adjacent_indices, :, :]
        center_direction = x["center_direction"][:, center_indices]
        adjacent_directions = x["adjacent_direction"][:, adjacent_indices]

        sample = {
            "adjacent_img": adjacent_images,
            "adjacent_directions": adjacent_directions,
            # create channel dimension
            "center_img": center_image.unsqueeze(0),
            "center_directions": center_direction,
            "subject_id": x["subject_id"],
            "slice_id": x["slice_id"],
            "n_direction": x["n_direction"],
        }

        return sample


class GetCenterAdjacentImg(MapTransform):
    def __init__(self, keys):
        MapTransform.__init__(self, keys)

    def __call__(self, x: Dict):
        center_indices, adjacent_indices = sample_directions_training(
            x["adjacent_direction"],
            x["center_direction"],
            int(x["n_neighbor_directions"]),
        )
        center_image = x["center_img"][center_indices, :, :]
        adjacent_images = x["adjacent_img"][adjacent_indices, :, :]
        center_direction = x["center_direction"][:, center_indices]
        adjacent_directions = x["adjacent_direction"][:, adjacent_indices]

        sample = {
            "adjacent_img": adjacent_images,
            "adjacent_directions": adjacent_directions,
            # create channel dimension
            "center_img": center_image.unsqueeze(0),
            "center_directions": center_direction,
            "subject_id": x["subject_id"],
            "slice_id": x["slice_id"],
        }

        return sample


def get_train_data_transform() -> Compose:
    return Compose(
        [
            LoadImaged(
                keys=["img", "mask"],
            ),
            EnsureChannelFirstd(keys=["img", "mask"]),
            Orientationd(keys=["img", "mask"], axcodes="RAS"),
            Spacingd(
                keys=["img", "mask"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            CenterSpatialCropd(
                keys=["img", "mask"],
                roi_size=[196, 196, 196],
            ),
            SpatialPadd(
                keys=["img", "mask"],
                spatial_size=[196, 196, 196],
            ),
            ScaleIntensityRangePercentilesd(
                keys=[
                    "img",
                ],
                lower=0.5,
                upper=99.5,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            ToTensord(
                keys=["img", "label"],
            ),
        ]
    )
