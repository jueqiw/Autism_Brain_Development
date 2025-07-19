import sys
import random
from typing import Tuple

import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from cvxopt import solvers, matrix 
from sklearn.gaussian_process.kernels import  RBF 

# with help from ChatGPT and DeepSeek


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(
        self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None
    ):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [
            torch.exp(-L2_distance / bandwidth_temp)
            for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
        )
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss

# taken from https://github.com/MaterialsInformaticsDemo/MK-MMD/blob/main/code/MK_MMD.py
class MKMMD():
    def __init__(self, gamma_list=[2,1,1/2,1/4,1/8,], kernel_num = 5):
        '''
        Our code is designed for educational purposes, 
        and to make it easier to understand, 
        we have implemented only the RBF (Radial Basis Function) kernel.
        
        This case focuses on solving the weights of kernels. 
        The estimation of length scales is crucial in kernel-based models,
        For further details on the method (length scales), please visit the following link: 
        [https://github.com/MaterialsInformaticsDemo/DAN/blob/main/code/MK_MMD.py].

        :param gamma_list: list of length scales for rbf kernels
        :param kernel_num: number of kernels in MK_MMD
        '''
        if len(gamma_list) != kernel_num: 
            print('please assign specific length scales for each rbf kernel')
        self.kernel_num = kernel_num
        kernel_list = []
        for i in range(kernel_num):
            kernel_list.append(RBF(gamma_list[i],"fixed"))
        self.kernel_list = kernel_list

    def predict(self, Xs, Xt,) :
        '''
        :param Xs: ns * m_feature, source domain data 
        :param Xt: nt * m_feature, target domain data

        return :
        the result of MK_MMD & weights of kernels 
        '''
        # cal weights for each rbf kernel
        # two rows above section 2.2 Empirical estimate of the MMD, asymptotic distribution, and test
        h_matrix = [] # 5 * 5 
        for i in range(self.kernel_num):
            _, h_k_vector = funs(Xs, Xt, self.kernel_list[i], MMD = False, h_k_vector = True)
            h_matrix.append(h_k_vector)
        h_matrix = np.vstack(h_matrix)
        print('h matrix is calculated')

        # cal the covariance matrix of h_matrix
        # Eq.(7)
        Q_k = np.cov(h_matrix)
        # cal the weights of kernels, Eq.(11)
        # vector η_k, Eq.(2)
        η_k = []
        for k in range(self.kernel_num):
            MMD, _ = funs(Xs, Xt, self.kernel_list[k], MMD = True, h_k_vector = False)
            η_k.append(MMD)
        print('η_k is calculated')

        # solve the standard quadratic programming problem 
        # see : https://github.com/Bin-Cao/KMMTransferRegressor/blob/main/KMMTR/KMM.py
        P = 2 * matrix(Q_k + 1e-5 * np.eye(self.kernel_num)) # λm = 1e-5 
        # q = - η_k ， maximum η_k * beta in QB
        q = matrix(-np.array(η_k).reshape(-1,1))
        G = matrix(-np.eye(self.kernel_num))
        # the summation of the beta is 1, Eq.(3), let's D = 1
        A = matrix(np.ones((1,self.kernel_num)))
        b=matrix(1.)
        h=matrix(np.zeros((self.kernel_num,1)))
        # P is 5 * 5
        # q is 5 * 1
        # G is 5 * 5
        # A is 1 * 5
        # b = 1, h = 5*1
        solvers.options['show_progress'] = False
        sol = solvers.qp(P,q,G,h,A,b)
        beta = sol['x']
        print('the optimal weights are found')
        MK_MMD = np.array(η_k) @ np.array(beta)

        kernel = beta[0] *  self.kernel_list[0]
        for k in range(self.kernel_num - 1):
            kernel += beta[k+1] * self.kernel_list[k+1]
            
        return MK_MMD, np.array(beta), kernel
        
def funs(Xs, Xt, kernel, MMD = True, h_k_vector = False):
    if MMD == True:
        # cal MMD for one rbf kernel
        # Eq.(1) in paper
        dim = np.array(Xs).ndim
        Xs = np.array(Xs).reshape(-1,dim)
        Xt = np.array(Xt).reshape(-1,dim)
        EXX_= kernel(Xs,Xs)
        EYY_= kernel(Xt,Xt)
        EYX_= kernel(Xt,Xs)
        EXY_= kernel(Xs,Xt)
        MMD = np.array(EXX_).mean() + np.array(EYY_).mean() - np.array(EYX_).mean() - np.array(EXY_).mean()
    else: 
        MMD = None
        pass

    if h_k_vector == True:
        # cal vector h_k(x,x',y,y'), contains m**2*n**2 terms
        # between Eq.(1) and Eq.(2)
        # k(x, x') is the element of matrix EXX_
        # k(y, y') is the element of matrix EYY_
        # k(x, y') and k(x', y) are the element of matrix EXY_
        ns, nt = len(Xs), len(Xt)
        combin_ns = generate_combinations(ns)
        combin_nt = generate_combinations(nt)
        h_k_vector = []
        for x in range(len(combin_ns)):
            for y in range(len(combin_nt)):
                S_x = np.array(Xs[combin_ns[x][0]]).reshape(-1,1) # x
                S_x_ =  np.array(Xs[combin_ns[x][1]]).reshape(-1,1) # x'
                T_x =  np.array(Xt[combin_nt[y][0]]).reshape(-1,1) # y
                T_x_ =  np.array(Xt[combin_nt[y][1]]).reshape(-1,1) # y'
                h_k = kernel(S_x,S_x_) + kernel(T_x,T_x_) - kernel(S_x,T_x_) - kernel(S_x_,T_x)
                h_k_vector.append(h_k[0][0])
        h_k_vector = np.array(h_k_vector)
    else: 
        h_k_vector = None
        pass
    return MMD, h_k_vector

def generate_combinations(n):
    # Cn^2
    combinations = []
    for i in range(n):
        for j in range(i, n):
            combinations.append((i, j))
    return combinations


def groupwise_attention_loss(attention_matrices, group_labels, k=10, margin=1.0):
    """
    Group-wise loss for cross-attention matrices.

    Args:
        attention_matrices (torch.Tensor): Cross-attention matrices of shape (B, n, m).
        group_labels (torch.Tensor): Tensor of shape (B,) containing group labels (0=control, 1=ASD).
        k (int): Number of top values to consider in the attention matrix.
        margin (float): Margin for dissimilarity between group means.

    Returns:
        torch.Tensor: Computed loss value.
    """
    # Separate attention matrices by group
    asd_attention = attention_matrices[group_labels == 1]
    control_attention = attention_matrices[group_labels == 0]

    # Compute group means
    asd_mean = asd_attention.mean(dim=0)
    control_mean = control_attention.mean(dim=0)

    # Compute similarity loss within each group
    asd_loss = torch.norm(asd_attention - asd_mean, dim=(1, 2)).mean()
    control_loss = torch.norm(control_attention - control_mean, dim=(1, 2)).mean()

    # Compute dissimilarity loss between groups
    inter_group_loss = max(0, margin - torch.norm(asd_mean - control_mean))

    # Combine losses
    total_loss = asd_loss + control_loss + inter_group_loss
    return total_loss


# Code is adpated from: https://github.com/wangz10/contrastive_loss/blob/master/losses.py#L50
def pdist_euclidean(z):
    """
    Computes pairwise Euclidean distance matrix.
    """
    dist_matrix = torch.cdist(z, z, p=2)
    return dist_matrix


def square_to_vec(D):
    """
    Converts a square distance matrix to a vector form.
    """
    return D[torch.triu_indices(D.size(0), D.size(1), offset=1).unbind()]


def get_contrast_batch_labels(y):
    """
    Generates contrastive labels for pairs.
    """
    if y.dim() > 1:
        y = y.squeeze()  # Ensure y is 1D

    y_i, y_j = torch.meshgrid(y, y, indexing="ij")
    return (y_i == y_j).float()[
        torch.triu_indices(y.size(0), y.size(0), offset=1).unbind()
    ]


def max_margin_contrastive_loss(
    z, y, margin=1.0, metric="euclidean", if_matrix=False, tau=0.07
):
    z = z.view(z.size(0), -1)  # Flatten to [bsz, n_features * m_features]
    y = y.view(-1)

    if metric == "euclidean":
        D = pdist_euclidean(z)
    elif metric == "cosine":
        D = 1 - torch.mm(F.normalize(z, p=2, dim=1), F.normalize(z, p=2, dim=1).T) / tau
    else:
        raise ValueError("Unsupported metric")

    d_vec = square_to_vec(D)
    y_contrasts = get_contrast_batch_labels(y)

    loss = (
        y_contrasts * d_vec.pow(2) + (1 - y_contrasts) * F.relu(margin - d_vec).pow(2)
    ).mean()

    return loss


# some code is modified from https://github.com/rongzhou7/MCLCA/blob/main/MCLCA.py#L9
def full_contrastive_loss(z_alpha, z_beta, tau=0.07, lambda_param=0.5):
    """
    Compute the full contrastive loss considering all negative samples explicitly,
    without normalizing by batch size.
    """
    # Normalize the embedding vectors
    z_alpha_norm = F.normalize(z_alpha, p=2, dim=1)
    z_beta_norm = F.normalize(z_beta, p=2, dim=1)

    # Calculate the cosine similarity matrix
    sim_matrix = torch.mm(z_alpha_norm, z_beta_norm.t()) / tau
    # Extract similarities of positive pairs (same index pairs)
    positive_examples = torch.diag(sim_matrix)
    # Apply exponential to the similarity matrix for negative pairs handling
    exp_sim_matrix = torch.exp(sim_matrix)
    # Create a mask to zero out positive pair contributions in negative pairs sum
    mask = torch.eye(z_alpha.size(0)).to(z_alpha.device)
    exp_sim_matrix -= mask * exp_sim_matrix
    # Sum up the exponentiated similarities for negative pairs
    negative_sum = torch.sum(exp_sim_matrix, dim=1)

    # Calculate the contrastive loss for one direction (alpha as anchor)
    L_alpha_beta = -torch.sum(torch.log(positive_examples / negative_sum))

    # Repeat the steps for the other direction (beta as anchor)
    sim_matrix_T = sim_matrix.t()
    positive_examples_T = torch.diag(sim_matrix_T)
    exp_sim_matrix_T = torch.exp(sim_matrix_T)
    exp_sim_matrix_T -= mask * exp_sim_matrix_T
    negative_sum_T = torch.sum(exp_sim_matrix_T, dim=1)
    L_beta_alpha = -torch.sum(torch.log(positive_examples_T / negative_sum_T))

    # Combine the losses from both directions, balanced by lambda
    loss = lambda_param * L_alpha_beta + (1 - lambda_param) * L_beta_alpha
    return loss  # Return the unnormalized total loss


def contrastive_loss(z_alpha, z_beta, lambda_param, tau=0.07):
    """
        Compute the contrastive loss L_cont(α, β) for two sets of embeddings.

        Parameters:
        - z_alpha: Embeddings from modality α, tensor of shape (batch_size, embedding_size)
        - z_beta: Embeddings from modality β, tensor of shape (batch_size, embedding_size)
        - tau: Temperature parameter for scaling the cosine similarity
        - lambda_param: Weighting parameter to balance the loss terms
    x
        Returns:
        - loss: The computed contrastive loss
    """

    # Compute the cosine similarity matrix
    sim_matrix = (
        torch.mm(F.normalize(z_alpha, p=2, dim=1), F.normalize(z_beta, p=2, dim=1).t())
        / tau
    )
    # Diagonal elements are positive examples
    positive_examples = torch.diag(sim_matrix)
    # Compute the log-sum-exp for the denominator
    sum_exp = torch.logsumexp(sim_matrix, dim=1)

    # Loss for one direction (α anchoring and contrasting β)
    L_alpha_beta = -torch.mean(positive_examples - sum_exp)

    # Loss for the other direction (β anchoring and contrasting α)
    L_beta_alpha = -torch.mean(
        torch.diag(
            torch.mm(
                F.normalize(z_beta, p=2, dim=1), F.normalize(z_alpha, p=2, dim=1).t()
            )
            / tau
        )
        - torch.logsumexp(sim_matrix.t(), dim=1)
    )

    # Combined loss, balanced by lambda
    loss = lambda_param * L_alpha_beta + (1 - lambda_param) * L_beta_alpha

    return loss
