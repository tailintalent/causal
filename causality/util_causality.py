
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from sklearn.model_selection import train_test_split

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from causal.pytorch_net.util import to_Variable, to_np_array, expand_tensor
from causal.datasets.prepare_dataset import process_time_series
from causal.util import format_list, normalize_tensor
import causal.estimators.entropy_estimators as ee


# In[2]:


def get_prfs(causality_truth, causality_pred):
    assert causality_pred.shape == causality_truth.shape
    assert len(causality_pred.shape) == 2
    from sklearn.metrics import precision_recall_fscore_support as prfs
    precision, recall, F1, _ = prfs(to_np_array(causality_truth).flatten(), to_np_array(causality_pred).flatten())
    return precision[1], recall[1], F1[1]


def get_ROC_AUC(causality_truth, causality_value, neglect_idx = None):
    from sklearn.metrics import roc_auc_score
    causality_truth = to_np_array(causality_truth)
    causality_value = to_np_array(causality_value)
    assert not np.isinf(causality_value).any() and not np.isnan(causality_value).any()
    ROC_AUC = roc_auc_score(flatten_matrix(causality_truth, neglect_idx), flatten_matrix(causality_value, neglect_idx))
    ROC_AUC_list = []
    for i in range(len(causality_truth)):
        try:
            if neglect_idx is not None:
                ROC_AUC_ele = roc_auc_score(flatten_matrix(causality_truth[i], neglect_idx = i), flatten_matrix(causality_value[i], neglect_idx = i))
            else:
                ROC_AUC_ele = roc_auc_score(flatten_matrix(causality_truth[i]), flatten_matrix(causality_value[i]))
        except:
            ROC_AUC_ele = np.NaN
        ROC_AUC_list.append(ROC_AUC_ele)
    ROC_AUC_mean = np.nanmean(ROC_AUC_list)
    return ROC_AUC, ROC_AUC_mean, ROC_AUC_list


def get_PR_AUC(causality_truth, causality_value, neglect_idx = None):
    from sklearn.metrics import precision_recall_curve, auc
    causality_truth = to_np_array(causality_truth)
    causality_value = to_np_array(causality_value)
    assert not np.isinf(causality_value).any() and not np.isnan(causality_value).any()
    precision_curve, recall_curve, _ = precision_recall_curve(flatten_matrix(causality_truth, neglect_idx), flatten_matrix(causality_value, neglect_idx))
    PR_AUC = auc(recall_curve, precision_curve)
    PR_AUC_list = []
    for i in range(len(causality_truth)):
        try:
            if neglect_idx is not None:
                precision_curve_ele, recall_curve_ele, _ = precision_recall_curve(flatten_matrix(causality_truth[i], neglect_idx = i), flatten_matrix(causality_value[i], neglect_idx = i))
            else:
                precision_curve_ele, recall_curve_ele, _ = precision_recall_curve(flatten_matrix(causality_truth[i]), flatten_matrix(causality_value[i]))
            PR_AUC_ele = auc(recall_curve_ele, precision_curve_ele)
        except:
            PR_AUC_ele = np.NaN
        PR_AUC_list.append(PR_AUC_ele)
    PR_AUC_mean = np.nanmean(PR_AUC_list)
    return PR_AUC, PR_AUC_mean, PR_AUC_list


def get_AUCs(causality_value, A_whole, neglect_idx = None, verbose = True):
    result = {}
    if A_whole is not None:
        causality_truth = np.abs(A_whole) > 0
        if len(causality_truth.shape) == 3:
            causality_truth = causality_truth.any(-2)
        ROC_AUC, ROC_AUC_mean, ROC_AUC_list = get_ROC_AUC(causality_truth, causality_value, neglect_idx = neglect_idx)
        PR_AUC, PR_AUC_mean, PR_AUC_list = get_PR_AUC(causality_truth, causality_value, neglect_idx = neglect_idx)
        if verbose:
            print("ROC_AUC: {0:.6f}\tROC_AUC_mean: {1:.6f}\tROC_AUC_list: {2}".format(ROC_AUC, ROC_AUC_mean, format_list(ROC_AUC_list, decimals = 4)))
            print("PR_AUC:  {0:.6f}\tPR_AUC_mean:  {1:.6f}\tPR_AUC_list:  {2}".format(PR_AUC, PR_AUC_mean, format_list(PR_AUC_list, decimals = 4)))
        result["ROC_AUC_dict"] = {"ROC_AUC": ROC_AUC, "ROC_AUC_mean": ROC_AUC_mean, "ROC_AUC_list": ROC_AUC_list}
        result["PR_AUC_dict"] = {"PR_AUC": PR_AUC, "PR_AUC_mean": PR_AUC_mean, "PR_AUC_list": PR_AUC_list}
        result["A_whole"] = A_whole
    return result


def get_MIs(
    X,
    y,
    noise_amp_all,
    group_sizes,
    mode = "xn-y",
    noise_type = "uniform-series",
    estimate_method = "k-nearest",
    ):
    assert len(X.shape) == len(y.shape) == 3
    _, K, N = X.shape
    if isinstance(group_sizes, int):
        num_models = int(N / group_sizes)
    else:
        num_models = len(group_sizes)
    num = noise_amp_all.size(0)
    if noise_type == "uniform-series":
        MI = np.zeros((num, num_models))
    elif noise_type == "fully-random":
        MI = np.zeros((num, K, num_models))
    else:
        raise

    X_std = X.std(0)
    is_cuda = X.is_cuda
    device = torch.device("cuda" if is_cuda else "cpu")

    if noise_type == "uniform-series":  
        for i in range(num):
            noise_amp_core = X_std * expand_tensor(noise_amp_all[i].to(device), -1, group_sizes)
            X_tilde = X + torch.randn(X.size()).to(device) * noise_amp_core
            if mode == "xn-y":
                arg1 = y
                arg2 = X_tilde
            elif mode == "x-y":
                arg1 = y
                arg2 = X
            elif mode == "xn-x":
                arg1 = X_tilde
                arg2 = X
            else:
                raise

            for j in range(num_models):
                if mode == "xn-x":
                    if estimate_method == "k-nearest":
                        MI[i, j] = ee.mi(to_np_array(arg1[:, :, j * group_sizes : (j + 1) * group_sizes].contiguous().view(arg1.size(0), -1)), to_np_array(arg2[:, :, j * group_sizes : (j + 1) * group_sizes].contiguous().view(arg2.size(0), -1)))
                    elif estimate_method == "Gaussian":
                        entropy_X_tilde = get_entropy_Gaussian(arg1[:, :, j * group_sizes : (j + 1) * group_sizes].contiguous().view(arg1.size(0), -1), is_diagonal = False)
                        KM = group_sizes * K
                        entropy_noise = (KM / float(2) * np.log2(2 * np.pi * np.e) + torch.log2(noise_amp_core[:, j]).sum())
                        MI[i, j] = entropy_X_tilde - entropy_noise
                    else:
                        raise
                else:
                    if estimate_method == "k-nearest":
                        MI[i, j] = ee.mi(to_np_array(arg1[:, :, i * group_sizes : (i + 1) * group_sizes].contiguous().view(arg1.size(0), -1)), to_np_array(arg2[:, :, j * group_sizes : (j + 1) * group_sizes].contiguous().view(arg2.size(0), -1)))
                    elif estimate_method == "Gaussian":
                        MI[i, j] = get_entropy_Gaussian(arg1[:, :, i * group_sizes : (i + 1) * group_sizes].contiguous().view(arg1.size(0), -1), arg2[:, :, j * group_sizes : (j + 1) * group_sizes].contiguous().view(arg2.size(0), -1))
                    else:
                        raise
    elif noise_type == "fully-random":
        for i in range(num):
            noise_amp_core = X_std * noise_amp_all[i].to(device)
            X_tilde = X + torch.randn(X.size()).to(device) * noise_amp_core
            if mode == "xn-y":
                arg1 = y
                arg2 = X_tilde
            elif mode == "x-y":
                arg1 = y
                arg2 = X
            elif mode == "xn-x":
                arg1 = X_tilde
                arg2 = X
            else:
                raise

            for k in range(K):
                for j in range(num_models):
                    if mode == "xn-x":
                        MI[i, k, j] = ee.mi(to_np_array(arg1[:, k, j * group_sizes : (j + 1) * group_sizes]), to_np_array(arg2[:, k, j * group_sizes : (j + 1) * group_sizes]))
                    else:
                        MI[i, k, j] = ee.mi(to_np_array(arg1[:, :, i * group_sizes : (i + 1) * group_sizes].contiguous().view(arg1.size(0), -1)), to_np_array(arg2[:, k, j * group_sizes : (j + 1) * group_sizes]))
    else:
        raise
    return MI


def flatten_matrix(matrix, neglect_idx = None):
    if neglect_idx is None:
        return matrix.flatten()
    elif neglect_idx == "diagonal":
        rows, columns = matrix.shape
        indices = np.array(list(set(range(matrix.size)) - {i * columns + i for i in range(rows)}))
        return matrix.flatten()[indices]
    elif isinstance(neglect_idx, int):
        indices = list(range(neglect_idx)) + list(range(neglect_idx + 1, matrix.size))
        return matrix.flatten()[indices]
    else:
        raise


def plot_clusters(noise_amp, isplot = True):
    hist = noise_amp.view(-1,1).cpu().detach().numpy()
    kmean = KMeans(init='k-means++', n_clusters=2, n_init=10)
    labels = kmean.fit_predict(hist)
    if isplot:
        plt.figure(figsize = (8, 6))
        if (labels == 0).sum() > 0:
            plt.hist(hist[labels==0], bins = 15)
        if (labels == 1).sum() > 0:
            plt.hist(hist[labels==1], bins = 15)
        plt.xlabel("eta values")
        plt.ylabel("counts")
        plt.title("Clustering of eta matrix element values")
    
    if (labels == 1).sum() > 0:
        if hist[labels == 1].max() > hist[labels == 0].max():
            higher_idx = 1
            lower_idx = 0
        else:
            higher_idx = 0
            lower_idx = 1
        max_lower = hist[labels == lower_idx].max()
        min_higher = hist[labels == higher_idx].min()
        threshold = (max_lower + min_higher) / 2
    else:
        threshold = hist.mean()
    if isplot:
        plt.axvline(threshold, color = "k", linewidth = 1)
        plt.show()
    return float(threshold)



def plot_examples(X_ele, y_ele, title = None, isplot = True):
    if isplot:
        assert len(X_ele.shape) == len(y_ele.shape) == 3
        N = X_ele.size(-1)
        fig = plt.figure(figsize = (N * 6, 5))
        for j in range(N):
            plt.subplot(1, N, j + 1)
            plt.plot(np.concatenate((to_np_array(X_ele)[:,:,j].T, to_np_array(y_ele)[:,:,j].T), 0))
            if title is not None:
                plt.title(title)
        plt.show()


def parse_filename(filename):
    parse_dict = {}
    filename_split = filename.split("_")
    parse_dict["data_type"] = filename_split[0]
    parse_dict["method"] = filename_split[filename_split.index("method") + 1]
    parse_dict["N"] = int(filename_split[filename_split.index("N") + 1])
    parse_dict["true_id"] = int(filename_split[filename_split.index("id") + 1])
    parse_dict["K"] = int(filename_split[filename_split.index("K") + 1])
    parse_dict["K2"] = int(filename_split[filename_split.index("K2") + 1])
    parse_dict["velocity_type"] = filename_split[filename_split.index("vel") + 1]
    parse_dict["normalize"] = filename_split[filename_split.index("nor") + 1]
    parse_dict["split_mode"] = filename_split[filename_split.index("split") + 1]
    parse_dict["reg_amp"] = float(filename_split[filename_split.index("reg") + 1])
    parse_dict["lr"] = float(filename_split[filename_split.index("lr") + 1])
    parse_dict["loss_core"] = filename_split[filename_split.index("core") + 1]
    parse_dict["num_examples"] = int(filename_split[filename_split.index("num") + 1])
    if "struct" in filename_split:
        parse_dict["struct_tuple"] = filename_split[filename_split.index("struct") + 1]
    if "act" in filename_split:
        parse_dict["activation"] = filename_split[filename_split.index("act") + 1]
    if "idas" in filename_split:
        parse_dict["assigned_target_id"] = eval(filename_split[filename_split.index("idas") + 1])
    parse_dict["seed"] = int(filename_split[filename_split.index("seed") + 1])
    parse_dict["idx"] = filename_split[filename_split.index("idx") + 1][:-2]
    return parse_dict


def partition(X, group_sizes):
    """Partition the input according to the group_sizes at the last dimension"""
    List = []
    for i in range(X.size(-1) // group_sizes):
        List.append(X[..., i * group_sizes : (i + 1) * group_sizes])
    return List


def get_shapes(X, y, group_sizes, assigned_target_id):
    """Get required shapes from data"""
    _, K, N = X.shape
    _, K2, Ny = y.shape
    assert assigned_target_id is None or (assigned_target_id is not None and isinstance(assigned_target_id, int))
    if isinstance(group_sizes, int):
        num_models = int(N / group_sizes)
    else:
        num_models = len(group_sizes)
    if N != Ny:
        num_models_y = 1
        group_sizes_y = Ny
    else:
        num_models_y = num_models
        group_sizes_y = group_sizes
    
    if assigned_target_id is None:
        training_target_ids = list(range(num_models_y))
    else:
        training_target_ids = [assigned_target_id]
    return N, Ny, K, K2, num_models, num_models_y, group_sizes_y, training_target_ids