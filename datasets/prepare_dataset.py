
# coding: utf-8

# In[1]:


import numpy as np
from scipy.integrate import odeint
import torch
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), "..", ".."))
from causal.util import plot_matrices, normalize_tensor
from causal.pytorch_net.util import get_activation, to_Variable, to_np_array


# ## Synthetic datasets:

# In[2]:


def process_time_series(time_series, K, K2 = 1, velocity_type = "pos", normalize = 0, is_cuda = False):
    X = []
    y = []
    if normalize == 4:
        num_time_series, length, N = time_series.shape
        time_series = time_series.view(-1, N)
        time_series = (time_series - time_series.mean(0, keepdim = True)) / time_series.std(0, keepdim = True)
        time_series = time_series.view(num_time_series, length, N)
    if velocity_type == "both":
        for i in range(len(time_series)):
            X_ele = []
            y_ele = []
            for k in range(1, time_series.size(1) - K - K2 + 1):
                combined_X = torch.stack([time_series[i, k : k + K], time_series[i, k : k + K] - time_series[i, k - 1 : k + K - 1]], -1)
                combined_X = combined_X.view(*combined_X.size()[:-2], -1)
                combined_y = torch.stack([time_series[i, k + K : k + K + K2], time_series[i, k + K : k + K + K2] - time_series[i, k + K -1 : k + K + K2 - 1]], -1)
                combined_y = combined_y.view(*combined_y.size()[:-2], -1)
                X_ele.append(combined_X)
                y_ele.append(combined_y)
            X.append(torch.stack(X_ele))
            y.append(torch.stack(y_ele))
    elif velocity_type == "pos":
        for i in range(len(time_series)):
            X_ele = []
            y_ele = []
            for k in range(1, time_series.size(1) - K - K2 + 1):
                X_ele.append(time_series[i, k : k + K])
                y_ele.append(time_series[i, k + K : k + K + K2])
            X.append(torch.stack(X_ele))
            y.append(torch.stack(y_ele))
    elif velocity_type == "velocity":
        for i in range(len(time_series)):
            X_ele = []
            y_ele = []
            for k in range(1, time_series.size(1) - K - K2 + 1):
                X_ele.append(time_series[i, k : k + K] - time_series[i, k - 1 : k + K - 1])
                y_ele.append(time_series[i, k + K : k + K + K2] - time_series[i, k + K -1 : k + K + K2 - 1])
            X.append(torch.stack(X_ele))
            y.append(torch.stack(y_ele))
    else:
        raise
    X = torch.stack(X)
    y = torch.stack(y)
    X, y = normalize_tensor(X, y, normalize = normalize)
    if is_cuda:
        X = X.cuda()
        y = y.cuda()
    return X, y


def get_A_matrix(N, K, p_N = 0.5, p_K = 0.5, dist_type = "uniform", isTorch = True, is_cuda = False, isplot = False):
    on_series = np.random.choice([0, 1], size = N, p = [1-p_N, p_N])
    A = np.zeros((K, N))
    for j, element in enumerate(on_series):
        if element == 1:
            if dist_type == "uniform":
                A[:, j] = (np.random.randn(K) * 0.1 + 0.4) * np.random.choice([1,-1], size = K) * np.random.choice([0, 1], size = K, p = [1-p_K, p_K])
            elif dist_type == "lognormal":
                A[:, j] = np.random.lognormal(size = K) * np.random.choice([1,-1], size = K) * np.random.choice([0, 1], size = K, p = [1-p_K, p_K])
            elif dist_type == "exponential":
                A[:, j] = np.random.exponential(size = K) * np.random.choice([1,-1], size = K) * np.random.choice([0, 1], size = K, p = [1-p_K, p_K])
            else:
                raise Exception("dist_type {0} not valid!".format(dist_type))
    if isplot:
        plot_matrices([np.abs(A)])
    if isTorch:
        A = torch.Tensor(A)
        if is_cuda:
            A = A.cuda()
    return A


def get_synthetic_data(
    N,
    K,
    K2 = 1,
    p_N = 0.5,
    p_K = 0.5,
    time_length = 10,
    num_examples = 3000,
    A_whole = None,
    dist_type = "uniform",
    mode = "linear",
    indi_activation = None,
    noise_multiplicative_matrix = None,
    noise_variance = None,
    observational_model = None,
    isplot = False,
    velocity_type = "pos",
    normalize = 0,
    is_cuda = False,
    ):
    # Configure causal factor A:
    if A_whole is None:
        A_whole = np.zeros((N, K, N))
        for n in range(N):
            A_whole[n] = get_A_matrix(N, K, p_N = p_N, p_K = p_K, dist_type = dist_type, isTorch = False)
    else:
        assert A_whole.shape[0] == A_whole.shape[2], "A_whole must have the shape of (N, K, N)!"
    if isplot:
        plot_matrices(A_whole)

    # Configure individual scaling factor B:
    if indi_activation is not None:
        B_whole = np.random.choice([1,-1], size = (N, K, N)) * (np.random.rand(N, K, N) + 1)
    else:
        B_whole = None

    # configure noise_multiplicative_matrix
    if noise_multiplicative_matrix is not None:
        if isinstance(noise_multiplicative_matrix, str) and noise_multiplicative_matrix == "random":
            noise_multiplicative_matrix = torch.zeros(N, K, N)
            for n in range(N):
                noise_multiplicative_matrix[n] = get_A_matrix(N, K, p_N = p_N, p_K = p_K, dist_type = dist_type, isTorch = True).abs()
        else:
            noise_multiplicative_matrix = to_Variable(noise_multiplicative_matrix).abs()
            shape = noise_multiplicative_matrix.shape
            if len(shape) == 2:
                assert shape[0] == shape[1]
                noise_multiplicative_matrix = noise_multiplicative_matrix.unsqueeze(1).repeat(1, K, 1)
        if isplot:
            print("noise_multiplicative_matrix:")
            plot_matrices(to_np_array(noise_multiplicative_matrix))

    # Begin time-series generation:
    time_series_all = []
    for i in range(int(num_examples / time_length)):
        x = torch.randn(K, N)
        time_series = [x]
        for j in range(time_length):
            if noise_variance is not None:
                if isinstance(noise_variance, np.ndarray) or isinstance(noise_variance, list):
                    noise_to_add = np.random.multivariate_normal(np.zeros(N), noise_variance, size = time_length)
                else:
                    noise_to_add = np.random.randn(time_length, N) * np.sqrt(noise_variance)
            x_t = torch.zeros(1, N)
            for n in range(N):
                if indi_activation is not None:
                    x_core = get_activation(indi_activation)(torch.FloatTensor(B_whole[n]) * x)
                else:
                    x_core = x
                if noise_multiplicative_matrix is None:
                    x_t[0, n] = get_activation(mode)((torch.FloatTensor(A_whole[n]) * x_core).sum())
                else:
                    x_t[0, n] = get_activation(mode)((torch.FloatTensor(A_whole[n]) * noise_multiplicative_matrix[n] * torch.randn(K, N) * x_core + 
                                                      torch.FloatTensor(A_whole[n]) * (noise_multiplicative_matrix[n] < 1e-9).float() * x_core).sum())
                if noise_variance is not None:
                    x_t[0, n] += noise_to_add[j, n]
            time_series.append(x_t)
            x = torch.cat([x[1:], x_t], 0)
        time_series = torch.cat(time_series, 0)
        time_series_all.append(time_series)

    time_series_all = torch.stack(time_series_all)

    if isplot:
        for kk in range(4):
            plt.figure(figsize = (20, 8))
            plt.plot(time_series_all[kk].numpy())
            plt.legend(list(range(time_series_all.size(-1))))

    # Apply observational model if given:
    if observational_model is not None:
        if isinstance(observational_model, list):
            shape = time_series_all.shape
            time_series_all_new = []
            for i, model in enumerate(observational_model):
                time_series = model(time_series_all[...,i:i+1].contiguous().view(-1, 1))
                time_series_all_new.append(time_series)
            time_series_all = torch.cat(time_series_all_new, -1).view(shape[0], shape[1], -1)
        else:
            time_series_all = observational_model(time_series_all)
        time_series_all = to_Variable(time_series_all, requires_grad = False)

    X, y = process_time_series(time_series_all, K = K, K2 = K2, velocity_type = velocity_type, normalize = normalize, is_cuda = is_cuda)
    if is_cuda:
        time_series_all = time_series_all.cuda()
    return (X, y), A_whole, B_whole, time_series_all

