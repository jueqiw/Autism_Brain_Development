import sys
from argparse import ArgumentParser, Namespace

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PathwayEncoderDataset(Dataset):
    def __init__(
        self,
        pathway: pd.DataFrame,
        label: pd.DataFrame,
        hparams: Namespace,
    ):
        self.pathway = pathway
        self.label = label.to_numpy()
        self.datast = hparams.dataset

    def __len__(self):
        return len(self.pathway)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pathway = self.pathway.iloc[idx, :]
        label = np.expand_dims(self.label[idx], axis=0)
        sample = {
            "pathway": torch.tensor(pathway).float(),
            "label": torch.tensor(label).float(),
        }

        return sample


# "Volume (Cortical Parcellation) of",
# "Surface Area of",
# "Cortical Thickness Average of",
# "Cortical Thickness Standard Deviation of",


# create datalaoder in pytorch to load csv files
class BrainPathwayDataset(Dataset):
    def __init__(
        self,
        img: pd.DataFrame,
        pathway: pd.DataFrame,
        label: pd.DataFrame,
        hparams: Namespace,
    ):
        self.img = img
        self.pathway = pathway
        self.label = label.to_numpy()
        self.datast = hparams.dataset

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # create a channel dimension for img feature
        img = np.expand_dims(self.img[idx, :], axis=0)

        if self.datast == "ADNI":
            # take every 4th column
            img_volumn = img[:, 0::4]
            img_suface = img[:, 1::4]
            img_thickness_avg = img[:, 2::4]
            img_thickness_std = img[:, 3::4]
        if self.datast == "ACE":
            img_volumn = img[:, :210]
            img_suface = img[:, 210:420]
            img_thickness_avg = img[:, 420:630]
            img_thickness_std = img[:, 630:840]

        img = np.concatenate(
            [img_volumn, img_suface, img_thickness_avg, img_thickness_std], axis=0
        )

        pathway = self.pathway.iloc[idx, :]
        label = np.expand_dims(self.label[idx], axis=0)
        sample = {
            "img": torch.tensor(img).float(),
            "pathway": torch.tensor(pathway).float(),
            "label": torch.tensor(label).float(),
        }

        return sample
