
# coding: utf-8

# In[1]:


import numpy as np
import torch

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from causal.pytorch_net.util import to_Variable, to_np_array, flatten, expand_tensor


# In[2]:


"""Obtain entropy and MI estimators assuming the data follows a multivariate Gaussian distribution"""

def get_cov_np(X, y = None):
    """Obtain covariate matrix from numpy input"""
    X = flatten(X)
    if y is not None:
        y = flatten(y)
        X = np.concatenate([X, y], -1)
    num_examples, dim = X.shape
    X = X - X.mean(0, keepdims = True)
    cov = np.dot(X.T, X) / num_examples
    return cov


def get_MI_from_cov_np(cov, sub_size = None):
    """Calculate the MI between sub-components using the Gaussian cov matrix in numpy"""
    if sub_size is None:
        sub_size = int(cov.shape[0] / 2)
    det1 = np.linalg.det(cov[:sub_size, :sub_size])
    det2 = np.linalg.det(cov[sub_size:, sub_size:])
    det = np.linalg.det(cov)
    MI = np.log(det1 * det2 / det) / 2 / np.log(2)
    return MI


def get_cov(X, y = None):
    """Obtain covariate matrix from input"""
    X = flatten(X)
    if y is not None:
        y = flatten(y)
        X = torch.cat([X, y], -1)
    num_examples, dim = X.shape
    X = X - X.mean(0, keepdim = True)
    cov = torch.matmul(X.transpose(0,1), X) / num_examples
    return cov


def get_entropy_from_cov(cov):
    """Calculate the differential entropy using the Gaussian cov matrix"""
    assert cov.shape[0] == cov.shape[1]
    size = cov.shape[0]
    entropy = (size / float(2) * np.log2(2 * np.pi * np.e) + torch.log2(torch.det(cov)) / 2)
    return entropy


def get_MI_from_cov(cov, sub_size = None):
    """Calculate the MI between sub-components using the Gaussian cov matrix"""
    if sub_size is None:
        sub_size = int(cov.shape[0] / 2)
    det1 = torch.det(cov[:sub_size, :sub_size])
    det2 = torch.det(cov[sub_size:, sub_size:])
    det = torch.det(cov)
    MI = torch.log2(det1 * det2 / det) / 2
    return MI


def get_entropy_Gaussian(X, is_diagonal = False):
    """Calculate the differential entropy regarding the input as multivariate-Gaussian"""
    cov = get_cov(X)
    if is_diagonal:
        cov = cov.diag() * torch.eye(cov.shape[0])
    return get_entropy_from_cov(cov)


def get_entropy_Gaussian_list(X_list, is_diagonal = False):
    """Calculate a list of differential entropy regarding each input as multivariate-Gaussian"""
    List = []
    for X in X_list:
        List.append(get_entropy_Gaussian(X, is_diagonal = is_diagonal))
    return torch.stack(List)


def get_MI_Gaussian(X, y):
    """Calculate the mutual information regarding the input as multivariate-Gaussian"""
    cov = get_cov(X, y)
    assert len(X.shape) == 2
    subsize = X.shape[-1]
    return get_MI_from_cov(cov)


def get_noise_entropy(noise_amp_core, K, group_sizes):
    """Obtain the entropy of Gaussian noise"""
    entropy_noise_list = []
    for i in range(noise_amp_core.shape[-1] // group_sizes):
        noise_amp_core_ele = noise_amp_core[..., i * group_sizes: (i + 1) * group_sizes]
        KM = group_sizes * K
        entropy_noise = (KM / float(2) * np.log2(2 * np.pi * np.e) + torch.log2(noise_amp_core_ele).sum())
        entropy_noise_list.append(entropy_noise)
    entropy_noise_list = torch.stack(entropy_noise_list)
    return entropy_noise_list

