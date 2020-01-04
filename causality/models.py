
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import datetime
import random
from copy import deepcopy
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torch.utils.data as data_utils
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from causal.settings.filepath import cau_PATH
import causal.estimators.entropy_estimators as ee
from causal.causality.util_causality import get_AUCs, plot_clusters, plot_examples, get_ROC_AUC, get_PR_AUC, partition
from causal.datasets.prepare_dataset import process_time_series, get_synthetic_data
from causal.pytorch_net.net import MLP, Multi_MLP, LSTM, Model_Ensemble, Mixture_Gaussian, Model_with_uncertainty, train, load_model_dict
from causal.pytorch_net.util import to_Variable, to_np_array, get_optimizer, get_activation, Loss_with_uncertainty, get_criterion, shrink_tensor, expand_tensor, permute_dim, flatten, fill_triangular, matrix_diag_transform, to_string
from causal.util import train_test_split, Early_Stopping, record_data, make_dir, plot_matrices, norm, format_list, get_args


# In[ ]:


class MLP_noise(MLP):
    def __init__(
        self,
        input_size,
        struct_param,
        W_init_list = None,     # initialization for weights
        b_init_list = None,     # initialization for bias
        settings = {},          # Default settings for each layer, if the settings for the layer is not provided in struct_param
        is_cuda = False,
        ):
        super(MLP_noise, self).__init__(input_size = input_size,
                                        struct_param = struct_param,
                                        W_init_list = W_init_list,
                                        b_init_list = b_init_list,
                                        settings = settings,
                                        is_cuda = is_cuda
                                       )
        self.device = torch.device("cuda" if self.is_cuda else "cpu")

    def forward(
        self,
        input,
        noise_amp = None,
        added_noise_type = "Gaussian",
        mix_coeff = None,
        permute_idx = None,
        permute_mode = "permute",
        group_sizes = None,
        **kwargs
        ):
        if noise_amp is not None:
            assert mix_coeff is None
            if added_noise_type == "Gaussian":
                input = input + torch.randn(input.size()).to(self.device) * noise_amp
            elif added_noise_type == "uniform":
                input = input + (torch.rand(input.size()) - 0.5).to(self.device) * noise_amp
            else:
                raise
        if mix_coeff is not None:
            mix_coeff = mix_coeff.clamp(0, 1)
            input_permuted = permute_dim(input, 2, permute_idx, group_sizes, mode = permute_mode)
            input = input * (1 - mix_coeff) + input_permuted * mix_coeff
        output = input
        for k in range(len(self.struct_param)):
            output = getattr(self, "layer_{0}".format(k))(output)
        return output

    
class Variational_Entropy(nn.Module):
    def __init__(
        self,
        num_models,
        num_components,
        dim,
        is_cuda = False,
        ):
        """Variational upper bound for differential entropy"""
        super(Variational_Entropy, self).__init__()
        self.num_models = num_models
        self.num_components = num_components
        self.dim = dim
        self.is_cuda = is_cuda
        self.device = torch.device("cuda" if self.is_cuda else "cpu")
        for i in range(self.num_models):
            setattr(self, "model_{0}".format(i), Mixture_Gaussian(num_components = num_components,
                                                                  dim = dim,
                                                                  is_cuda = is_cuda,
                                                                 ))

    def initialize_optim(self, lr = 1e-3):
        for i in range(self.num_models):
            setattr(self, "optimizer_{0}".format(i), get_optimizer("adam", lr = lr, parameters = getattr(self, "model_{0}".format(i)).parameters()))

    
    def initialize(self, input_list, num_samples = 10):
        for i in range(self.num_models):
            print("\nmodel_{0}:".format(i))
            getattr(self, "model_{0}".format(i)).initialize(flatten(input_list[i]),
                                                            num_samples = num_samples,
                                                            verbose = True,
                                                           )
        self.initialize_optim()


    def get_entropy(self, input, noise_amp_core, group_sizes):
        entropy_list = []
        X_list = partition(input, group_sizes)
        noise_amp_core_list = partition(noise_amp_core, group_sizes)
        for i in range(self.num_models):
            X = X_list[i]
            X_tilde = X + torch.randn(X.size()).to(self.device) * noise_amp_core_list[i]
            entropy = getattr(self, "model_{0}".format(i)).get_loss(flatten(X_tilde))
            entropy_list.append(entropy)
        entropy_list = torch.stack(entropy_list)
        return entropy_list


    def train(self, input_list, num_steps = 10):
        for k in range(num_steps):
            loss_list = []
            for i, X in enumerate(input_list):
                getattr(self, "optimizer_{0}".format(i)).zero_grad()
                loss = getattr(self, "model_{0}".format(i)).get_loss(flatten(X).detach())
                loss_list.append(loss)
                loss.backward()
                getattr(self, "optimizer_{0}".format(i)).step()
            loss_list = torch.stack(loss_list)
            print("{0}: losses: {1}".format(k, to_string(to_np_array(loss_list), connect = "    ", num_digits = 4, num_strings = None)))

