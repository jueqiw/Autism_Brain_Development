"""
Code is modified from: https://github.com/SergioLeonardoMendes/normative_psychiatry/blob/main/src/python/models/vqvae.py
VQ-VAE model [1]

References:
    [1] - Neural Discrete Representation Learning (https://arxiv.org/pdf/1711.00937.pdf)
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast


class ResidualLayer(nn.Sequential):
    def __init__(self, n_channels, n_res_channels, p_dropout):
        super().__init__(
            nn.Conv3d(n_channels, n_res_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Dropout3d(p_dropout),
            nn.Conv3d(n_res_channels, n_channels, kernel_size=1),
        )

    def forward(self, x):
        return F.relu(x + super().forward(x), True)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        n_embed=8,
        embed_dim=64,
        n_alpha_channels=1,
        n_channels=64,
        n_res_channels=64,
        n_res_layers=2,
        p_dropout=0.1,
        latent_resolution="low",
    ):
        super().__init__()
        self.n_embed = n_embed
        self.latent_resolution = latent_resolution

        if latent_resolution == "low":
            self.encoder = nn.Sequential(
                nn.Conv3d(n_alpha_channels, n_channels // 2, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Sequential(
                    *[
                        ResidualLayer(n_channels // 2, n_res_channels // 2, p_dropout)
                        for _ in range(n_res_layers)
                    ]
                ),
                nn.Conv3d(n_channels // 2, n_channels // 2, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Sequential(
                    *[
                        ResidualLayer(n_channels // 2, n_res_channels // 2, p_dropout)
                        for _ in range(n_res_layers)
                    ]
                ),
                nn.Conv3d(n_channels // 2, n_channels, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Sequential(
                    *[
                        ResidualLayer(n_channels, n_res_channels, p_dropout)
                        for _ in range(n_res_layers)
                    ]
                ),
                nn.Conv3d(n_channels, embed_dim, 3, stride=1, padding=1),
            )

            self.decoder = nn.Sequential(
                nn.Conv3d(embed_dim, n_channels, 3, stride=1, padding=1),
                nn.Sequential(
                    *[
                        ResidualLayer(n_channels, n_res_channels, p_dropout)
                        for _ in range(n_res_layers)
                    ]
                ),
                nn.ConvTranspose3d(n_channels, n_channels // 2, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Sequential(
                    *[
                        ResidualLayer(n_channels // 2, n_res_channels // 2, p_dropout)
                        for _ in range(n_res_layers)
                    ]
                ),
                nn.ConvTranspose3d(
                    n_channels // 2, n_channels // 2, 4, stride=2, padding=1
                ),
                nn.ReLU(),
                nn.Sequential(
                    *[
                        ResidualLayer(n_channels // 2, n_res_channels // 2, p_dropout)
                        for _ in range(n_res_layers)
                    ]
                ),
                nn.ConvTranspose3d(
                    n_channels // 2, n_alpha_channels, 4, stride=2, padding=1
                ),
            )

        elif latent_resolution == "mid":
            self.encoder = nn.Sequential(
                nn.Conv3d(n_alpha_channels, n_channels // 2, 6, stride=3, padding=2),
                nn.ReLU(),
                nn.Sequential(
                    *[
                        ResidualLayer(n_channels // 2, n_res_channels // 2, p_dropout)
                        for _ in range(n_res_layers)
                    ]
                ),
                nn.Conv3d(n_channels // 2, n_channels, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Sequential(
                    *[
                        ResidualLayer(n_channels, n_res_channels, p_dropout)
                        for _ in range(n_res_layers)
                    ]
                ),
                nn.Conv3d(n_channels, embed_dim, 3, stride=1, padding=1),
            )

            self.decoder = nn.Sequential(
                nn.Conv3d(embed_dim, n_channels, 3, stride=1, padding=1),
                nn.Sequential(
                    *[
                        ResidualLayer(n_channels, n_res_channels, p_dropout)
                        for _ in range(n_res_layers)
                    ]
                ),
                nn.ConvTranspose3d(n_channels, n_channels // 2, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Sequential(
                    *[
                        ResidualLayer(n_channels // 2, n_res_channels // 2, p_dropout)
                        for _ in range(n_res_layers)
                    ]
                ),
                nn.ConstantPad3d((1, 0, 1, 0, 1, 0), 0),
                nn.ConvTranspose3d(
                    n_channels // 2, n_alpha_channels, 6, stride=3, padding=3
                ),
            )

    def forward(self, x):
        with autocast(enabled=True):
            z = self.encoder(x)
            x_tilde = self.decoder(z)

        return x_tilde
