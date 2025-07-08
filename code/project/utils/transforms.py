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
)


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
            # ScaleIntensityRangePercentilesd(
            #     keys=[
            #         "img",
            #     ],
            #     lower=0.5,
            #     upper=99.5,
            #     b_min=0.0,
            #     b_max=1.0,
            #     clip=True,
            # ),
            ToTensord(
                keys=["img", "mask"],
            ),
        ]
    )
