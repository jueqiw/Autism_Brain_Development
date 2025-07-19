# from x_transformers import CrossAttender
import sys
import math

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

from models.sparsemax import Sparsemax


class PathwayEncoder(nn.Module):
    def __init__(
        self,
        n_pathway,
        classifier_latnet_dim,
        normalization: str,
        hidden_dim_qk: int,
        hidden_dim_q: int,
        hidden_dim_k: int,
        hidden_dim_v: int,
        relu_at_coattention: bool,
        gamma: float = 1.0,
        soft_sign_constant: float = 0.5,
    ):
        super(PathwayEncoder, self).__init__()
        self.n_pathway = n_pathway
        self.hidden_dim_q = hidden_dim_q
        self.hidden_dim_qk = hidden_dim_qk
        self.hidden_dim_k = hidden_dim_k
        self.hidden_dim_v = hidden_dim_v
        self.hidden_dim = 1
        self.n_heads = 1
        self.query_d = 1
        self.key_value_d = 4
        self.classifier_latnet_dim = classifier_latnet_dim
        self.normalization = normalization
        self.relu_at_coattention = relu_at_coattention
        self.gamma = gamma
        self.variance_reduction_factor = self.gamma * (self.n_pathway**0.5 / 2)
        self.soft_sign_constant = soft_sign_constant

        # cross attention
        self.query_weights = nn.Parameter(torch.randn(self.query_d, self.hidden_dim_q))
        self.query_weights_2 = nn.Parameter(
            torch.randn(self.hidden_dim_q, self.hidden_dim_qk)
        )
        self.query_weights_3 = nn.Parameter(
            torch.randn(self.hidden_dim_qk, self.hidden_dim)
        )
        self.query_relu = nn.LeakyReLU(0.2)
        self.key_weights = nn.Parameter(
            torch.randn(self.key_value_d, self.hidden_dim_qk)
        )
        self.key_weights_2 = nn.Parameter(
            torch.randn(self.hidden_dim_k, self.hidden_dim_qk)
        )
        self.key_relu = nn.LeakyReLU(0.2)
        self.value_weights = nn.Parameter(
            torch.randn(self.key_value_d, self.hidden_dim_v)
        )
        self.value_weights_2 = nn.Parameter(
            torch.randn(self.hidden_dim_v, self.hidden_dim)
        )
        self.value_relu = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(0.5)
        self.relu_1 = nn.LeakyReLU(0.2)
        self.classifier_1 = nn.Linear(self.n_pathway, self.classifier_latnet_dim)
        self.dropout_2 = nn.Dropout(0.5)
        self.relu_2 = nn.LeakyReLU(0.2)
        self.classifier_2 = nn.Linear(self.classifier_latnet_dim, 1)
        self.sparsemax = Sparsemax(dim=-1)

        if self.normalization == "batch":
            self.query_norm = nn.BatchNorm1d(self.hidden_dim_q)
            self.query_norm_2 = nn.BatchNorm1d(self.hidden_dim_qk)
            self.query_norm_3 = nn.BatchNorm1d(self.hidden_dim_qk)
            self.key_norm = nn.BatchNorm1d(self.hidden_dim_qk)
            self.value_norm = nn.BatchNorm1d(self.hidden_dim_v)
            self.norm_1 = nn.BatchNorm1d(self.n_pathway)
            self.norm_2 = nn.BatchNorm1d(self.classifier_latnet_dim)
        elif self.normalization == "layer":
            self.query_norm = nn.LayerNorm(self.hidden_dim_qk)
            self.query_norm_2 = nn.LayerNorm(self.hidden_dim_qk)
            self.query_norm_3 = nn.LayerNorm(self.hidden_dim_qk)
            self.key_norm = nn.LayerNorm(self.hidden_dim_qk)
            self.value_norm = nn.LayerNorm(self.hidden_dim_v)
            self.norm_1 = nn.LayerNorm(self.n_pathway)
            self.norm_2 = nn.LayerNorm(self.classifier_latnet_dim)
        if self.normalization == "None":
            self.norm_1 = nn.BatchNorm1d(self.n_pathway)
            self.norm_2 = nn.BatchNorm1d(self.classifier_latnet_dim)

        # initialize the weights
        nn.init.xavier_uniform_(self.query_weights)
        nn.init.xavier_uniform_(self.query_weights_2)
        nn.init.xavier_uniform_(self.query_weights_3)
        nn.init.xavier_uniform_(self.key_weights)
        nn.init.xavier_uniform_(self.value_weights)
        nn.init.xavier_uniform_(self.classifier_1.weight)
        nn.init.zeros_(self.classifier_1.bias)

    def encoder(self, modality):
        latent = torch.matmul(modality, self.query_weights)
        if self.normalization == "batch":
            batch_size = latent.shape[0]
            latent = latent.reshape(-1, self.hidden_dim_q)
            latent = self.query_norm(latent)
            latent = latent.view(batch_size, self.n_pathway, self.hidden_dim_q)
        elif self.normalization == "layer":
            latent = self.query_norm(latent)
        if self.relu_at_coattention:
            latent = self.query_relu(latent)
        latent = torch.matmul(latent, self.query_weights_2)

        if self.normalization == "batch":
            batch_size = latent.shape[0]
            latent = latent.reshape(-1, self.hidden_dim_qk)
            latent = self.query_norm_2(latent)
            latent = latent.view(batch_size, self.n_pathway, self.hidden_dim_qk)
        elif self.normalization == "layer":
            latent = self.query_norm_2(latent)
        if self.relu_at_coattention:
            latent = self.query_relu_2(latent)

        latent = torch.matmul(latent, self.query_weights_3)
        return latent

    def forward(self, modality):
        # add a last dimension
        # transpose the last two dimensions
        modality = modality.unsqueeze(-1)  # (batch_size, n_pathway, 1)
        out = self.encoder(modality)
        out_squeeze = out.squeeze(-1)
        # query_matrix = query_matrix.squeeze(-1)
        # entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
        # out_squeeze = torch.cat([out_squeeze, query_matrix], dim=-1)
        # check whether out_squeeze contain 0 when it does the normalization
        out_squeeze = self.norm_1(out_squeeze)
        combined = self.dropout_1(self.relu_1(out_squeeze))
        classifier_1 = self.classifier_1(combined)
        classifier_1 = self.norm_2(classifier_1)
        classifier_1 = self.dropout_2(self.relu_2(classifier_1))
        logits = self.classifier_2(classifier_1)
        return logits


class BrainPathwayAnalysis(nn.Module):
    def __init__(
        self,
        n_pathway,
        n_img_features,
        classifier_latnet_dim,
        normalization: str,
        hidden_dim_qk: int,
        hidden_dim_q: int,
        hidden_dim_k: int,
        hidden_dim_v: int,
        relu_at_coattention: bool,
        gamma: float = 1.0,
        soft_sign_constant: float = 0.5,
    ):
        super(BrainPathwayAnalysis, self).__init__()
        self.n_pathway = n_pathway
        self.n_img_features = n_img_features
        self.hidden_dim_q = hidden_dim_q
        self.hidden_dim_qk = hidden_dim_qk
        self.hidden_dim_k = hidden_dim_k
        self.hidden_dim_v = hidden_dim_v
        self.hidden_dim = 1
        self.n_heads = 1
        self.query_d = 1
        self.key_value_d = 4
        self.classifier_latnet_dim = classifier_latnet_dim
        self.normalization = normalization
        self.relu_at_coattention = relu_at_coattention
        self.gamma = gamma
        self.variance_reduction_factor = self.gamma * (self.n_pathway**0.5 / 2)
        self.soft_sign_constant = soft_sign_constant

        # cross attention
        self.query_weights = nn.Parameter(torch.randn(self.query_d, self.hidden_dim_q))
        self.query_weights_2 = nn.Parameter(
            torch.randn(self.hidden_dim_q, self.hidden_dim_qk)
        )
        self.query_relu = nn.LeakyReLU(0.2)
        self.key_weights = nn.Parameter(
            torch.randn(self.key_value_d, self.hidden_dim_qk)
        )
        self.key_weights_2 = nn.Parameter(
            torch.randn(self.hidden_dim_k, self.hidden_dim_qk)
        )
        self.key_relu = nn.LeakyReLU(0.2)
        self.value_weights = nn.Parameter(
            torch.randn(self.key_value_d, self.hidden_dim_v)
        )
        self.value_weights_2 = nn.Parameter(
            torch.randn(self.hidden_dim_v, self.hidden_dim)
        )
        self.value_relu = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(0.5)
        self.relu_1 = nn.LeakyReLU(0.2)
        self.classifier_1 = nn.Linear(self.n_pathway, self.classifier_latnet_dim)
        self.dropout_2 = nn.Dropout(0.5)
        self.relu_2 = nn.LeakyReLU(0.2)
        self.classifier_2 = nn.Linear(self.classifier_latnet_dim, 1)
        self.sparsemax = Sparsemax(dim=-1)

        if self.normalization == "batch":
            self.query_norm = nn.BatchNorm1d(self.hidden_dim_q)
            self.key_norm = nn.BatchNorm1d(self.hidden_dim_qk)
            self.value_norm = nn.BatchNorm1d(self.hidden_dim_v)
            self.norm_1 = nn.BatchNorm1d(self.n_pathway)
            self.norm_2 = nn.BatchNorm1d(self.classifier_latnet_dim)
        elif self.normalization == "layer":
            self.query_norm = nn.LayerNorm(self.hidden_dim_qk)
            self.key_norm = nn.LayerNorm(self.hidden_dim_qk)
            self.value_norm = nn.LayerNorm(self.hidden_dim_v)
            self.norm_1 = nn.LayerNorm(self.n_pathway)
            self.norm_2 = nn.LayerNorm(self.classifier_latnet_dim)
        if self.normalization == "None":
            self.norm_1 = nn.BatchNorm1d(self.n_pathway)
            self.norm_2 = nn.BatchNorm1d(self.classifier_latnet_dim)

        # initialize the weights
        nn.init.xavier_uniform_(self.query_weights)
        nn.init.xavier_uniform_(self.query_weights_2)
        nn.init.xavier_uniform_(self.key_weights)
        nn.init.xavier_uniform_(self.value_weights)
        nn.init.xavier_uniform_(self.classifier_1.weight)
        nn.init.zeros_(self.classifier_1.bias)

    def cross_attention(self, query, key, value):
        query_matrix = torch.matmul(query, self.query_weights)
        if self.normalization == "batch":
            batch_size = query_matrix.shape[0]
            query_matrix = query_matrix.reshape(-1, self.hidden_dim_q)
            query_matrix = self.query_norm(query_matrix)
            query_matrix = query_matrix.view(
                batch_size, self.n_pathway, self.hidden_dim_q
            )
        elif self.normalization == "layer":
            query_matrix = self.query_norm(query_matrix)
        if self.relu_at_coattention:
            query_matrix = self.query_relu(query_matrix)
        query_matrix = torch.matmul(query_matrix, self.query_weights_2)
        key_matrix = torch.matmul(key, self.key_weights)
        if self.normalization == "batch":
            batch_size = key_matrix.shape[0]
            key_matrix = key_matrix.reshape(-1, self.hidden_dim_qk)
            key_matrix = self.key_norm(key_matrix)
            key_matrix = key_matrix.view(
                batch_size, self.n_img_features, self.hidden_dim_qk
            )
        elif self.normalization == "layer":
            key_matrix = self.key_norm(key_matrix)
        if self.relu_at_coattention:
            query_matrix = self.key_relu(query_matrix)
        # key_matrix = torch.matmul(key_matrix, self.key_weights_2)
        value_matrix = torch.matmul(value, self.value_weights)
        if self.normalization == "batch":
            batch_size = value_matrix.shape[0]
            value_matrix = value_matrix.reshape(-1, self.hidden_dim_v)
            value_matrix = self.value_norm(value_matrix)
            value_matrix = value_matrix.view(
                batch_size, self.n_img_features, self.hidden_dim_v
            )
        elif self.normalization == "layer":
            value_matrix = self.value_norm(value_matrix)
        if self.relu_at_coattention:
            value_matrix = self.key_relu(value_matrix)
        value_matrix = torch.matmul(value_matrix, self.value_weights_2)
        atten_matrix = torch.matmul(query_matrix, key_matrix.transpose(-2, -1))
        atten_matrix = F.relu(atten_matrix)
        atten_matrix = atten_matrix / self.variance_reduction_factor
        # atten_matrix = self.softsign(atten_matrix)
        atten_matrix = atten_matrix / (atten_matrix.abs() + self.soft_sign_constant)
        # atten_matrix = torch.clamp(atten_matrix, min=0, max=1)
        # atten_matrix = self.sparsemax(atten_matrix)
        # atten_matrix = F.softmax(atten_matrix / math.sqrt(self.hidden_dim_qk), dim=-1)
        # check whether attention matrix contain nan
        out = torch.matmul(atten_matrix, value_matrix)
        return out, atten_matrix, query_matrix

    def forward(self, img, pathway):
        # add a last dimension
        # transpose the last two dimensions
        img = img.transpose(-2, -1)  # (batch_size, n_img, 4)
        pathway = pathway.unsqueeze(-1)  # (batch_size, n_pathway, 1)
        out, attn_weights, query_matrix = self.cross_attention(
            query=pathway, key=img, value=img
        )
        out_squeeze = out.squeeze(-1)
        # query_matrix = query_matrix.squeeze(-1)
        # entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
        # out_squeeze = torch.cat([out_squeeze, query_matrix], dim=-1)
        # check whether out_squeeze contain 0 when it does the normalization
        out_squeeze = self.norm_1(out_squeeze)
        combined = self.dropout_1(self.relu_1(out_squeeze))
        classifier_1 = self.classifier_1(combined)
        classifier_1 = self.norm_2(classifier_1)
        classifier_1 = self.dropout_2(self.relu_2(classifier_1))
        logits = self.classifier_2(classifier_1)
        return logits, attn_weights, out
