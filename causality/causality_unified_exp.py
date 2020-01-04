
# coding: utf-8

# In[ ]:


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
from causal.estimators.Gaussian_estimators import get_entropy_Gaussian, get_entropy_Gaussian_list, get_MI_Gaussian, get_noise_entropy
from causal.causality.models import MLP_noise, Variational_Entropy
from causal.causality.util_causality import get_AUCs, plot_clusters, plot_examples, get_ROC_AUC
from causal.causality.util_causality import get_PR_AUC, partition, get_shapes, get_MIs
from causal.datasets.prepare_dataset import process_time_series, get_synthetic_data
from causal.pytorch_net.net import MLP, Multi_MLP, LSTM, Model_Ensemble, Model_with_uncertainty, Mixture_Gaussian, train, load_model_dict
from causal.pytorch_net.util import plot_matrices, to_Variable, to_np_array, get_activation, Loss_with_uncertainty, get_optimizer, get_criterion, shrink_tensor, expand_tensor, permute_dim, flatten, fill_triangular, matrix_diag_transform, to_string, filter_filename, sort_two_lists
from causal.pytorch_net.logger import Logger
from causal.util import train_test_split, Early_Stopping, record_data, make_dir, norm, format_list, get_args

is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
isplot = False


# ## Settings:

# In[2]:


"""Data dimensions:
X_train, X_test has dimension of (#examples, K, N), where K is the maximum time horizons, (N // group_sizes) is the number of time series,
                                                    where group_sizes is the dimension for each time series.
y_train, y_test has dimension of (#examples, K2, N), where K2 is the number of time steps after the input X
"""
############################################################
# Choose one dataset:
############################################################
data_type = ("synthetic","softplus","tanh", 20, "lognormal", 1)    # Synthetic dataset (Section 4.1)
# data_type = ("breakout-CNN",)      # breakout dataset (Section 4.2)
# data_type = ("sleep-apnea",)         # heart rate vs. breath rate dataset (Section 4.3)
# data_type = ("ratEEG", "A")        # rat brain EEG dataset, normal (Section 4.3)
# data_type = ("ratEEG", "B")        # rat brain EEG dataset, after lesion (Section 4.3)

############################################################
# Choose one method:
############################################################
# method = ("MI", 5)
# method = ("trans-entropy", 3)
# method = ("G-linear", True)  # whether to fit intercept
# method = ("elasticNet",)
# method = ("causal-influence",0)
method = ("MPIR", "uniform-series", "Gaussian", 0.002, "info", "diag")

############################################################
# Settings:
############################################################
# Settings if using the Jupyter notebook to run:
exp_id = "exp1.0"
N = 10                        # Number of time-series variables
K = 3                         # Maximum time horizons for the input X
K2 = 1                        # number of time steps for y
id = 1                        # Legacy setting. Ineffective
velocity_type = "pos"         # if "pos", only use the time series itself. If "velocity", use the difference of time series. If "both", concatenate the above
normalize = 0                 # normalization type. 0 for no normalization, 2 for normalize to mean 0 and std of 1
split_mode = ("whole",)       # Legacy setting. Ineffective
reg_amp = 1e-4                # L1 regularization amplitude
noise_type = "uniform-series" # If "uniform-series", the relative noise amplitude parameter is shared across the same time series. If "fully-Gaussian", 
                              # different dimension and lag of the same time series have different noise amplitude parameters.
noise_mode = "additive"       # additive noise
lr = 1e-4                     # learning rate
group_sizes = 1               # group size for the data
loss_core = "mse"             # Legacy setting. Ineffective
num_examples = 10000          # number of examples
batch_size = 1000             # Batch size
struct_tuple = (8,8)          # Number of neurons for each hidden layer
activation = "leakyRelu"      # Activation for hidden layers
assigned_target_id = None     # If None, calculate causal matrix for all targets; if an integer, only calculate specific target
date_time = "{0}-{1}".format(datetime.datetime.now().month, datetime.datetime.now().day)
seed = 0                      # Seed
idx = "0"                     # id of experiment

# Settings if using the terminal to run:
exp_id = get_args(exp_id, 1)
data_type = get_args(data_type, 2, "tuple")
method = get_args(method, 3, "tuple")
N = get_args(N, 4, "int")
true_id = id = get_args(id, 5, "int")
K = get_args(K, 6, "int") 
K2 = get_args(K2, 7, "int")
velocity_type = get_args(velocity_type, 8)
normalize = get_args(normalize, 9, "int")
split_mode = get_args(split_mode, 10, "tuple")
reg_amp = get_args(reg_amp, 11, "float")
lr = get_args(lr, 12, "float")
loss_core = get_args(loss_core, 13)
num_examples = get_args(num_examples, 14, "int")
struct_tuple = get_args(struct_tuple, 15, "tuple")
activation = get_args(activation, 16)
assigned_target_id = get_args(assigned_target_id, 17, "eval")
date_time = get_args(date_time, 18)
seed = get_args(seed, 19, "int")
idx = get_args(idx, 20)

if method[0] == "MPIR":
    significance_test_mode = ("permute", "ratio", 0.5)   # Significance test setting.
else:
    significance_test_mode = None

# Seed:
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# For different methods use appropriate normalization of data:
if method[0] in ["elasticNet", "MI", "trans-entropy", "G-linear"]:
    normalize = 2
elif method[0] in ["MPIR", "causal-influence"]:
    normalize = 0
else:
    raise
if data_type[0] in ["breakout-CNN", "ratEEG", "sleep-apnea"]:
    normalize = 2
print("normalize = {0}".format(normalize))
neglect_idx = "diagonal"  # If "diagonal", neglect diagonal element when calculating AUC-ROC and AUC-PR. If None, include diagonal.


# ## Load dataset:

# In[3]:


test_size = 0.1
if data_type[0] == "synthetic":
    (X, y), A_whole, B_whole, time_series = get_synthetic_data(N = N, K = K, K2 = K2, p_N = 0.5, p_K = 0.5, time_length = data_type[3], dist_type = data_type[4], mode = data_type[1], 
                                                               indi_activation = data_type[2], num_examples = num_examples, noise_variance = data_type[5], velocity_type = velocity_type, normalize = normalize, is_cuda = is_cuda, isplot = isplot)

elif data_type[0] == "sleep-apnea":
    # Data from https://physionet.org/physiobank/database/santa-fe/
    # The three columns are heart rate, chest volume (respiration force) and blood oxygen concentration 
    file = "../datasets/sleep_apnea/b1.txt"
    time_series = []
    data_range = (2350, 3550)
    with open(file, "r") as f:
        for i, line in enumerate(f.readlines()):
            if data_range[0] < i < data_range[1]:
                time_series.append([float(element) for element in line.split(" ")[:3]])
        time_series = to_Variable(np.array(time_series), is_cuda = is_cuda).unsqueeze(0)        
    time_series = time_series[...,:2]
    time_series_plot = (time_series - time_series.mean(1, keepdim = True)) / time_series.std(1, keepdim = True)
    X, y = process_time_series(time_series, K = K, K2 = K2, velocity_type = velocity_type, normalize = normalize, is_cuda = is_cuda)
    if isplot:
        plt.figure(figsize = (20, 8))
        plt.plot(to_np_array(time_series_plot[0]))
        plt.legend(list(range(time_series_plot.size(-1))))
elif data_type[0] == "ratEEG":
    file = "../datasets/ratEEG/{0}-{1}.txt".format(data_type[0], data_type[1])
    time_series = []
    with open(file, "r") as f:
        for i, line in enumerate(f.readlines()):
            new_line = []
            for element in line.split(" "):
                if element != "":
                    if "\n" in element:
                        element = element[:-1]
                    new_line.append(eval(element))
            assert len(new_line) == 2
            time_series.append(new_line)
    time_series = to_Variable(np.array(time_series), is_cuda = is_cuda).unsqueeze(0)
    time_series_plot = time_series
    if data_type[1] == "A":
        time_series_plot[:,:,1] = time_series_plot[:,:,1] - 1
    else:
        time_series_plot[:,:,1] = time_series_plot[:,:,1] - 5  # 1: Left,  0: Right
    X, y = process_time_series(time_series, K = K, K2 = K2, velocity_type = velocity_type, normalize = normalize, is_cuda = is_cuda)
    if isplot:
        plt.figure(figsize = (20, 8))
        plt.plot(to_np_array(time_series_plot[0]))
        plt.legend(list(range(time_series_plot.size(-1))))
elif data_type[0][:8] == "breakout":
    time_series = to_Variable(pickle.load(open("../datasets/breakout/{0}.p".format(data_type[0]), "rb")), is_cuda = is_cuda)
    time_series = time_series[:int(num_examples / time_series.size(1))]
    X, y = process_time_series(time_series, K = K, K2 = K2, velocity_type = velocity_type, normalize = normalize, is_cuda = is_cuda)
else:
    raise Exception("data_type {0} not recognized!".format(data_type))
A_whole = A_whole if "A_whole" in locals() else None
N = X.size(-1)
if velocity_type == "both":
    group_sizes *= 2
    
# Append randomly permuted or constructed time series for significance test:
sig_dict = {}
if significance_test_mode is not None:
    sig_mode = significance_test_mode[0]
    if sig_mode == "permute":
        def append_sig_to_tensor(X, sig_idx, group_sizes):
            sig_X = X.view(-1, *X.shape[-2:])
            sig_X_perm = []
            for i in sig_idx:
                sig_X_perm.append(sig_X[torch.randperm(len(sig_X)), : , i * group_sizes: (i+1) * group_sizes])
            sig_X_perm = torch.cat(sig_X_perm, -1).view(*X.shape[:3], sig_num * group_sizes)
            X = torch.cat([X, sig_X_perm], -1)
            return X
        sig_mode_1 = significance_test_mode[1]
        if sig_mode_1 == "ratio":
            sig_ratio = significance_test_mode[2]
            sig_num = int(np.ceil(N * sig_ratio/ group_sizes))
            sig_num_y = int(np.ceil(y.shape[-1] * sig_ratio/ group_sizes))
        else:
            raise
        sig_idx = np.random.choice(int(N / group_sizes), sig_num, replace = False)
        X = append_sig_to_tensor(X, sig_idx, group_sizes)
        # Append y_append which is constructed to be caused by the first sig_num elements.
        sig_y_append = []
        control_matrix = ((1 + 0.1 * torch.randn(sig_num, K, K2)) / 3).to(device)
        for i in range(sig_num_y):
            sig_y_ele = torch.matmul(X[..., i * group_sizes: (i + 1) * group_sizes].transpose(-1, -2), control_matrix[i])
            sig_y_append.append(sig_y_ele)
        sig_y_append = torch.cat(sig_y_append, -2).transpose(-1, -2)
        y = torch.cat([y, sig_y_append], -1)
        
        # Record:
        sig_dict["sig_idx"] = sig_idx
        sig_dict["control_matrix"] = control_matrix
    else:
        raise

if split_mode[0] == "whole":
    (X_train, y_train), (X_test, y_test) = train_test_split(X, y, test_size = test_size)
elif split_mode[0] == "windowing":
    window_size = split_mode[1]
    (X_list, y_list), info_index = slide_windows(X, y, window_size = window_size, isplot = False)
    if False and isplot and velocity_type == "pos" and group_sizes == 2:
        show_gym_video(X_list, radius = 40)
else:
    raise


dirname = cau_PATH + "/{0}_{1}/".format(exp_id, date_time)
filename = dirname + "{0}_method_{1}_N_{2}_id_{3}_K_{4}_K2_{5}_vel_{6}_nor_{7}_split_{8}_reg_{9}_lr_{10}_core_{11}_num_{12}_struct_{13}_act_{14}_idas_{15}_seed_{16}_idx_{17}.p".format(
                        format_list(data_type, "-"), format_list(method, "-"), 
                        N, true_id, K, K2, velocity_type, normalize, format_list(split_mode, "-"), 
                        reg_amp, lr, loss_core, num_examples, format_list(struct_tuple, "-"), activation, assigned_target_id, seed, idx)
make_dir(filename)
print(filename)
window_idx = range(len(X_list)) if "X_list" in locals() else range(100)
causality_truth = np.abs(A_whole) > 0 if A_whole is not None else None


# ## Define methods:

# In[4]:


def get_mutual_information(X, y, group_sizes = 1, neighbors = 3, assigned_target_id = None, isplot = False):
    """Obtain pairwise mutual information"""
    N, Ny, K, K2, num_models, num_models_y, group_sizes_y, training_target_ids =         get_shapes(X, y, group_sizes, assigned_target_id)
    MI_matrix = np.zeros((num_models_y, num_models))

    for i in training_target_ids:
        for j in range(num_models):
            if isplot:
                print("target i: {0}\tsource j: {1}".format(i, j))
            y_chosen = y[:, :, i * group_sizes_y: (i + 1) * group_sizes_y]
            X_chosen = X[:, :, j * group_sizes: (j + 1) * group_sizes]
            MI_matrix[i, j] = ee.mi(y_chosen, X_chosen, k = neighbors)
            if isplot:
                plot_matrices([MI_matrix], images_per_row = 4)
    return MI_matrix, {}


def get_conditional_MI(X, y, group_sizes = 1, neighbors = 3, assigned_target_id = None):
    """Obtain transfer entropy using conditional mutual information"""
    N, Ny, K, K2, num_models, num_models_y, group_sizes_y, training_target_ids =         get_shapes(X, y, group_sizes, assigned_target_id)

    conditional_MI = np.zeros((num_models_y, num_models))
    for i in training_target_ids:
        for j in range(num_models):
            idx = torch.cat([torch.arange(j * group_sizes), torch.arange((j + 1) * group_sizes, N)]).long().to(device)
            y_chosen = y[:, :, i * group_sizes_y: (i + 1) * group_sizes_y]
            X_chosen = X[:, :, j * group_sizes: (j + 1) * group_sizes]
            X_conditioned = X.index_select(2, idx)
            conditional_MI[i, j] = ee.cmi(y_chosen, X_chosen, X_conditioned, k = neighbors)
    return conditional_MI, {}


def elastic_Net(
    X,
    y,
    group_sizes = 1,
    assigned_target_id = None,
    l1_ratio = [0.5, 0.8, 0.9, 0.95, 0.99],
    a = np.logspace(-4,-0.5,200),
    nfold = 5,
    tol = 1e-10,
    ):
    """Elastic Net fitting with cross-validation"""
    from sklearn import linear_model
    from sklearn.model_selection import TimeSeriesSplit
    N, Ny, K, K2, num_models, num_models_y, group_sizes_y, training_target_ids =         get_shapes(X, y, group_sizes, assigned_target_id)
    coeff_matrix = np.zeros((num_models_y, num_models))

    for i in training_target_ids:
        fold = TimeSeriesSplit(n_splits=nfold, max_train_size=None)
        model = linear_model.ElasticNetCV(l1_ratio, cv=fold, verbose=0, selection='random', tol=tol, alphas=a)
        #        
        model.fit(*to_np_array(*flatten(X, y[...,i * group_sizes : (i+1) * group_sizes])))
        coeff = np.abs(model.coef_.reshape(X.shape[1:])).mean(0)
        coeff_matrix[i] = coeff
    return coeff_matrix, {}


def get_Granger_linear_reg(X, y, group_sizes = 1, assigned_target_id = None, fit_intercept = False):
    """Obtain linear Granger causality using linear regression"""
    from sklearn.linear_model import LinearRegression
    N, Ny, K, K2, num_models, num_models_y, group_sizes_y, training_target_ids =         get_shapes(X, y, group_sizes, assigned_target_id)
    X, y = to_np_array(X, y)

    Granger_linear = np.zeros((num_models_y, num_models))
    error_matrix = np.zeros(num_models_y)
    error_matrix_ablated = np.zeros((num_models_y, num_models))
    for i in training_target_ids:
        model = LinearRegression(fit_intercept = fit_intercept)
        y_chosen = flatten(y[:,:, i * group_sizes_y: (i+1) * group_sizes_y])
        model.fit(flatten(X), y_chosen)
        pred = model.predict(flatten(X))
        error_matrix[i] = np.abs(pred - y_chosen).mean()

        for j in range(num_models):
            idx = np.concatenate([np.arange(j * group_sizes), np.arange((j + 1) * group_sizes, N)])
            X_ablated = np.take(X, indices = idx, axis = 2)
            model_minus_j = LinearRegression(fit_intercept = fit_intercept)
            model_minus_j.fit(flatten(X_ablated), y_chosen)
            pred = model_minus_j.predict(flatten(X_ablated))
            error_matrix_ablated[i, j] = np.abs(pred - y_chosen).mean()
            Granger_linear[i, j] = np.log(error_matrix_ablated[i, j] / error_matrix[i])
    return Granger_linear, {}


def get_model(input_size, struct_param, settings, is_uncertainty = False, loss_core = "mse", is_cuda = False):
    """Helper function for constructing the model"""
    if is_uncertainty:
        model_pred = MLP_noise(input_size = input_size,
                         struct_param = struct_param,
                         settings = settings,
                         is_cuda = is_cuda,
                        )
        model_logstd = MLP_noise(input_size = input_size,
                         struct_param = struct_param,
                         settings = settings,
                         is_cuda = is_cuda,
                        )
        model = Model_with_uncertainty(model_pred, model_logstd)
        criterion = Loss_with_uncertainty(core = loss_core)
        criterion_measure = get_criterion(loss_type = "get_Variance")
    else:
        model = MLP_noise(input_size = input_size,
                    struct_param = struct_param,
                    settings = settings,
                    is_cuda = is_cuda,
                   )
        criterion = criterion_measure = get_criterion(loss_type = loss_core)
    return model, criterion, criterion_measure


def get_causal_influence(
    X,
    y,
    validation_data,
    group_sizes = 1,
    average_times = 20,
    is_uncertainty = False,
    KL_estimator = "ee",
    causality_truth = None,
    isplot = True,
    verbose = True,
    noise_amp = None,
    permute_mode = "permute",
    **kwargs
    ):
    """Get causal_influence based on NN fitting
    The causal influence formula is given by Janzing, Dominik, et al. "Quantifying causal influences", 
    The Annals of Statistics 41.5 (2013): 2324-2358. The model is firstly learned from data.
    """
#     X = X_train
#     y = y_train
#     validation_data = (X_test, y_test)
#     KL_estimator = "ee"
#     kwargs = {}
#     causality_truth = None
#     is_uncertainty = False
#     isplot = True
#     average_times = 3
#     verbose = True

    X_valid, y_valid = validation_data
    struct_param = kwargs["struct_param"] if "struct_param" in kwargs else [[(K2, group_sizes), "Simple_Layer", {"activation": "linear"}]]
    settings = kwargs["settings"] if "settings" in kwargs else {}
    if "epochs" not in kwargs:
        kwargs["epochs"] = 15000   
    N, Ny, K, K2, num_models, num_models_y, group_sizes_y, training_target_ids =             get_shapes(X, y, group_sizes, assigned_target_id)
    
    input_size = (K, N)

    info_all = {}
    info_all["mse_full"] = - np.ones(num_models_y)
    info_all["mse_permuted"] = - np.ones((num_models_y, num_models, average_times))
    info_all["log_mse_ratio"] = - np.ones((num_models_y, num_models, average_times))
    info_all["causal_influence_all"] = np.zeros((num_models_y, num_models, average_times))
    info_all["causal_influence"] = np.zeros((num_models_y, num_models)) 

    if causality_truth is not None:
        if len(causality_truth.shape) == 3:
            causality_truth = causality_truth.any(-2, keepdims = True).squeeze()

    for i in training_target_ids:
        model, criterion, criterion_measure = get_model(input_size = input_size,
                                                        struct_param = struct_param,
                                                        settings = settings,
                                                        is_uncertainty = is_uncertainty,
                                                        loss_core = loss_core,
                                                        is_cuda = is_cuda,
                                                       )
        criterion_measure_full = deepcopy(criterion_measure)
        criterion_measure_full.reduce = False
        if noise_amp is None:
            _ = train(model, X, y[:,:, i*group_sizes_y: (i+1)*group_sizes_y], criterion = criterion, isplot = isplot, **kwargs)  
        else:
            _ = train(model, X, y[:,:, i*group_sizes_y: (i+1)*group_sizes_y], criterion = criterion, isplot = isplot, noise_amp = noise_amp, **kwargs)
        info_all["model_full_{0}".format(i)] = model.model_dict
        info_all["mse_full"][i] = to_np_array(model.get_loss(X_valid, y_valid[:,:, i*group_sizes_y: (i+1)*group_sizes_y], criterion_measure))
        if verbose:
            print("fitting full model {0},       mse = {1:.6f}".format(i, info_all["mse_full"][i]))
            try:
                sys.stdout.flush()
            except:
                pass
        for j in range(num_models):
            for k in range(average_times):
                print("j={0}\t k={1}".format(j,k))
                X_permuted = permute_dim(X, 2, j, group_sizes, mode = permute_mode)
                y_prime = model(X_permuted)
                concat_prev = torch.cat([X.view(X.size(0), -1), y[:,:, i*group_sizes_y: (i+1)*group_sizes_y].view(y.size(0), -1)], 1)
                concat_prime = torch.cat([X_permuted.view(X_permuted.size(0), -1), y_prime.view(y_prime.size(0), -1)], 1)
                if KL_estimator == "ee":
                    info_all["causal_influence_all"][i, j, k] = ee.kldiv(to_np_array(concat_prev), to_np_array(concat_prime))
                else:
                    raise

            info_all["causal_influence"][i, j] = info_all["causal_influence_all"][i,j].mean()
        if isplot == 2 or (isplot == 1 and j == num_models - 1):
            if causality_truth is not None:
                plot_matrices([info_all["causal_influence"], causality_truth], subtitles = ["causal_influence", "Truth"], images_per_row = 5)
            else:
                plot_matrices([info_all["causal_influence"]], subtitles = ["causal_influence"], images_per_row = 4)
        if A_whole is not None:
            try:
                get_AUCs(info_all["causal_influence"][:i+1], A_whole[:i+1], neglect_idx = neglect_idx)
            except Exception as e:
                print(e)
    return info_all["causal_influence"], info_all



class MPIR(object):
    """Our Causal Learning with Minimum Predictive information regularization (MPIR) method"""
    def __init__(
        self,
        ):
        pass
        
    
    def train(
        self,
        X,
        y,
        validation_data,
        group_sizes = 1,
        noise_type = "uniform-series",
        noise_loss_scale = 1e-2,
        norm_mode = "info",
        info_estimate_mode = "diag",
        is_uncertainty = False,
        A_whole = None,
        assigned_target_id = None,
        isplot = True,
        verbose = True,
        **kwargs
        ):
        # Obtain settings:
        device = torch.device("cuda" if is_cuda else "cpu")
        struct_param = kwargs["struct_param"] if "struct_param" in kwargs else [[10, "Simple_Layer", {"activation": "leakyRelu"}], [(K2, group_sizes), "Simple_Layer", {"activation": "linear"}]]
        settings = kwargs["settings"] if "settings" in kwargs else {}

        model_type = kwargs["model_type"] if "model_type" in kwargs else "MLP"
        loss_core = kwargs["loss_core"] if "loss_core" in kwargs else "mse"
        added_noise_type = kwargs["added_noise_type"] if "added_noise_type" in kwargs else "Gaussian"
        lr = kwargs["lr"] if "lr" in kwargs else 1e-4
        reg_amp = kwargs["reg_amp"] if "reg_amp" in kwargs else 1e-4
        batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 1000
        warmup = kwargs["warmup"] if "warmup" in kwargs else 400

        permute_mode = kwargs["permute_mode"] if "permute_mode" in kwargs else "permute"
        is_plot_MI = kwargs["is_plot_MI"] if "is_plot_MI" in kwargs else True
        num_samples = kwargs["num_samples"] if "num_samples" in kwargs else 1
        max_iter = kwargs["max_iter"] if "max_iter" in kwargs else 10000
        inspect_interval = kwargs["inspect_interval"] if "inspect_interval" in kwargs else 20
        plot_interval = kwargs["plot_interval"] if "plot_interval" in kwargs else 200
        save_interval = kwargs["save_interval"] if "save_interval" in kwargs else 1000
        patience = kwargs["patience"] if "patience" in kwargs else 40
        is_log = kwargs["is_log"] if "is_log" in kwargs else False
        record_mode = kwargs["record_mode"] if "record_mode" in kwargs else 0
        num_plots = 3

        # Obtain various shapes:
        N, Ny, K, K2, num_models, num_models_y, group_sizes_y, training_target_ids =             get_shapes(X, y, group_sizes, assigned_target_id)
        X_test, y_test = validation_data
        input_size = (K, N)
        self.info_dict = {}

        def get_X_tilde(X, X_std, noise_amp):
            noise_amp_core = X_std * expand_tensor(noise_amp, -1, group_sizes)
            return X + torch.randn(X.size()).to(device) * noise_amp_core

        self.data_record = {i: {} for i in range(num_models)}
        logger = Logger('./logs')

        if isplot:
            if "A_whole" in locals() and A_whole is not None:
                print("|A|:")
                if len(A_whole.squeeze().shape) > 2:
                    plot_matrices(np.abs(A_whole)[:num_plots], images_per_row = 3)
                else:
                    plot_matrices([np.abs(A_whole.squeeze())], images_per_row = 3)

                    
        # Initialize metrics:
        if noise_type == "uniform-series":
            self.noise_amp_all = torch.zeros(num_models_y, num_models)
        elif noise_type == "fully-random":
            self.noise_amp_all = torch.zeros(num_models_y, K, N)
        else:
            raise
        self.info_norm_all = torch.zeros(num_models_y, num_models)
        self.negative_log_noise_amp_all = torch.zeros(num_models_y, num_models)
        self.causality_pred_all = torch.zeros(num_models_y, num_models)

        X_std = X.std(0)
        # Main training loop:
        for target_id in training_target_ids:
            self.target_id = target_id
            print("Perform id: {0}".format(target_id))
            self.model, criterion, _ = get_model(input_size = input_size,
                                                struct_param = struct_param,
                                                settings = settings,
                                                is_uncertainty = False,
                                                loss_core = loss_core,
                                                is_cuda = is_cuda,
                                               )

            # Initialize noise_amp variable:
            if noise_type == "fully-random":
                noise_amp = torch.tensor((np.ones((K, N)) * 0.01).tolist(), requires_grad = True, device = device)
            elif noise_type == "uniform-series":
                noise_amp = torch.tensor((np.ones((1, int(N / group_sizes))) * 0.01).tolist(), requires_grad = True, device = device)
            else:
                raise Exception("noise_type {0} not recognized!".format(noise_type))    

            # If using variational upper bound for mutual information, first initialize the mixture Gaussians appropriately:
            if info_estimate_mode == "var":
                self.variational_entropy = Variational_Entropy(num_models = N // group_sizes,
                                                               num_components = 10,
                                                               dim = K * group_sizes,
                                                              )
                self.variational_entropy.initialize(partition(X, group_sizes), num_samples = 5)

            param_to_optimize = [{"params": self.model.parameters()}]
            param_to_optimize.append({"params": noise_amp})
            self.optimizer = optim.Adam(param_to_optimize, lr = lr)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = 0.2, patience = 10)
            self.early_stopping = Early_Stopping(patience = patience)

            dataset_train = data_utils.TensorDataset(X, y[:, :, target_id * group_sizes_y: (target_id + 1) * group_sizes_y])
            train_loader = data_utils.DataLoader(dataset_train, batch_size = batch_size, shuffle = True)
            to_stop = False

            # Train the model:
            for i in range(max_iter):
                if info_estimate_mode == "var" and i % 10 == 0 and i >= warmup:
                    print("training variational entropy estimator:")
                    for batch_id, (X_batch, y_batch) in enumerate(train_loader):
                        X_batch_new = get_X_tilde(X_batch, X_std, noise_amp)
                        self.variational_entropy.train(partition(X_batch_new, group_sizes), num_steps = 1)

                for batch_id, (X_batch, y_batch) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    reg = self.model.get_regularization(source = ["weight", "bias"], mode = "L1") * reg_amp
                    loss_train_total = torch.Tensor([0]).to(device)
                    for j in range(num_samples):
                        if noise_type == "uniform-series":
                            noise_amp_core = X_std * expand_tensor(noise_amp, -1, group_sizes)
                        else:
                            noise_amp_core = X_std * noise_amp
                        loss_train = self.model.get_loss(X_batch, y_batch, criterion, noise_amp = noise_amp_core, added_noise_type = added_noise_type)
                        loss_train_total = loss_train_total + loss_train
                    loss_train_total = loss_train_total / num_samples

                    loss_total = loss_train_total + reg * reg_amp
                    if i > warmup:
                        if noise_type == "uniform-series":
                            noise_amp_core = X_std * expand_tensor(noise_amp, -1, group_sizes)
                        else:
                            noise_amp_core = X_std * noise_amp

                        if info_estimate_mode == "diag":
                            if noise_type == "uniform-series":
                                info_estimate = norm(noise_amp, noise_mode = noise_mode, mode = norm_mode) * K * group_sizes
                            else:
                                info_estimate = norm(noise_amp, noise_mode = noise_mode, mode = norm_mode)
                        elif info_estimate_mode == "Gauss":
                            info_estimate = (get_entropy_Gaussian_list(partition(X_batch, group_sizes)) - get_noise_entropy(noise_amp_core, K, group_sizes)).sum()
                        elif info_estimate_mode == "var":
                            info_estimate = (self.variational_entropy.get_entropy(X_batch, noise_amp_core, group_sizes) - get_noise_entropy(noise_amp_core, K, group_sizes)).sum()
                        else:
                            raise
                        loss_total = loss_total + noise_loss_scale * info_estimate
                    loss_total.backward()
                    self.optimizer.step()

                # Validation and output:
                loss_test = self.model.get_loss(X_test, y_test[:, :, target_id * group_sizes_y : (target_id + 1) * group_sizes_y], criterion)
                if record_mode > 0:
                    record_data(self.data_record[target_id], [loss_train.item(), loss_test.item(), reg.item(), i], ["loss_train", "loss_test", "reg", "iter"])
                if i % inspect_interval == 0 or to_stop:
                    self.scheduler.step(loss_test.item())
                    to_stop = self.early_stopping.monitor(loss_test.item())
                    self.noise_amp_all[target_id] = noise_amp.abs()
                    print("iter {0}  \tloss_total: {1:.6f}\tloss_train: {2:.6f}\tloss_test: {3:.6f}\tnoise_norm: {4:.6f}\treg: {5:.9f}\tlr = {6:.6f}".format(
                        i, loss_total.item(), (loss_train_total / num_samples).item(), loss_test.item(), noise_loss_scale * norm(self.noise_amp_all[target_id], noise_mode = noise_mode, mode = norm_mode).item(), reg.item() * reg_amp, self.optimizer.param_groups[0]["lr"]), end = "")
                    try:
                        sys.stdout.flush()
                    except:
                        pass
                    print()
                    if is_log:
                        # Tensorboard Logging:
                        # 1. Log scalar values (scalar summary)
                        info = {'loss_train': loss_train.item(), 'loss_test': loss_test.item()}
                        for tag, value in info.items():
                            logger.log_scalar(tag, value, i+1)

                        # 2. Log values and gradients of the parameters (histogram summary)
                        for tag, value in self.model.named_parameters():
                            tag = tag.replace('.', '/')
                            logger.log_histogram(tag, value.data.cpu().numpy(), i+1)
                            logger.log_histogram(tag + '/grad', value.grad.data.cpu().numpy(), i+1)

                if i % plot_interval == 0:
                    if isplot:
                        plt.figure(figsize = (8,6))
                        plt.plot(self.data_record[target_id]["loss_train"], label = "loss_train")
                        plt.plot(self.data_record[target_id]["loss_test"], label = "loss_test")
                        plt.legend()
                        plt.show()
                        plt.figure(figsize = (8,6))
                        plt.semilogy(self.data_record[target_id]["loss_train"], label = "loss_train")
                        plt.semilogy(self.data_record[target_id]["loss_test"], label = "loss_test")
                        plt.legend()
                        plt.show()

                    if noise_type == "uniform-series":
                        self.info_norm_all[target_id] = torch.log2(1 + 1 / self.noise_amp_all[target_id] ** 2) / 2
                        self.negative_log_noise_amp_all[target_id] = -torch.log2(self.noise_amp_all[target_id])
                    else:
                        self.info_norm_all[target_id] = shrink_tensor((torch.log2(1 + 1 / self.noise_amp_all[target_id] ** 2) / 2).sum(-2), -1, group_sizes, "sum")
                        self.negative_log_noise_amp_all[target_id] = shrink_tensor(-torch.log2(self.noise_amp_all[target_id]).sum(-2), -1, group_sizes, "sum")
                    threshold = plot_clusters(self.info_norm_all[target_id: target_id + 1], isplot = isplot)
                    self.causality_pred_all[target_id] = (self.info_norm_all[target_id] > threshold).byte()

                    if A_whole is not None:
                        self.causality_truth_cal = np.abs(A_whole) > 0
                        self.causality_truth_cal = self.causality_truth_cal.any(-2, keepdims = True)

                        #Precision and recall:
                        try:
                            precision, recall, F1, _ = prfs(self.causality_truth_cal[:target_id + 1].flatten(), to_np_array(self.causality_pred_all[:target_id + 1]).flatten())
                            ROC_AUC = roc_auc_score(self.causality_truth_cal[:target_id + 1].flatten(), to_np_array(self.negative_log_noise_amp_all[:target_id + 1].view(-1)))
                            fpr, tpr, _ = roc_curve(self.causality_truth_cal[:target_id + 1].flatten(), to_np_array(self.negative_log_noise_amp_all[:target_id + 1].view(-1)))
                            precision_curve, recall_curve, _ = precision_recall_curve(self.causality_truth_cal[:target_id + 1].flatten(), to_np_array(self.negative_log_noise_amp_all[:target_id + 1].view(-1)))
                            PR_AUC = auc(recall_curve, precision_curve)
                        except:
                            pass

                        if isplot:
                            plt.figure(figsize = (8, 6))
                            plt.plot(fpr, tpr)
                            plt.plot([0,1], [0,1], "k--")
                            plt.xlabel("False Positive Rate")
                            plt.ylabel("True Positive Rate")
                            plt.xlim([0,1.05])
                            plt.ylim([0,1.05])
                            plt.title("ROC curve")
                            plt.show()

                            plt.figure(figsize = (8, 6))
                            plt.plot(precision_curve, recall_curve)
                            plt.xlabel("Precision")
                            plt.ylabel("Recall")
                            plt.xlim([0,1.05])
                            plt.ylim([0,1.05])
                            plt.title("Precision-Recall curve")
                            plt.show()

                            if i > 0:
                                plt.figure(figsize = (8, 6))
                                plt.plot(self.data_record[target_id]["iter_interval"], self.data_record[target_id]["precision"], label = "Precision")
                                plt.plot(self.data_record[target_id]["iter_interval"], self.data_record[target_id]["recall"], label = "Recall")
                                plt.plot(self.data_record[target_id]["iter_interval"], self.data_record[target_id]["F1"], label = "F1")
                                plt.plot(self.data_record[target_id]["iter_interval"], self.data_record[target_id]["ROC_AUC"], label = "ROC-AUC")
                                plt.plot(self.data_record[target_id]["iter_interval"], self.data_record[target_id]["PR_AUC"], label = "PR-AUC")
                                plt.legend()
                                plt.show()

                    if is_plot_MI:
                        MI_xn_x = get_MIs(X, y, noise_amp_all = self.noise_amp_all[:target_id + 1], group_sizes = group_sizes, mode = "xn-x", noise_type = noise_type)
                        MI_xn_y = get_MIs(X, y, noise_amp_all = self.noise_amp_all[:target_id + 1], group_sizes = group_sizes, mode = "xn-y", noise_type = noise_type)
                        MI_x_y = get_MIs(X, y, noise_amp_all = self.noise_amp_all[:target_id + 1], group_sizes = group_sizes, mode = "x-y", noise_type = noise_type)
                        if record_mode > 0:
                            record_data(self.data_record[target_id], [MI_xn_x, MI_xn_y, MI_x_y], ["MI_xn-x", "MI_xn-y", "MI_x-y"])

                    if isplot:
                        if num_models > 1 and noise_type == "fully-random":
                            plot_matrices(self.noise_amp_all, images_per_row = 5)
                        if A_whole is not None:
                            plot_matrices([to_np_array(self.info_norm_all), to_np_array(self.negative_log_noise_amp_all.squeeze()), to_np_array(self.causality_pred_all.squeeze()), self.causality_truth_cal.squeeze()], images_per_row = 5,
                                          subtitles = ["info-norm", "-log10(Eta)", "causality prediction", "true matrix"],
                                         )
                            if is_plot_MI:
                                plot_matrices([MI_xn_x, MI_xn_y, MI_x_y], images_per_row = 5, subtitles = ["MI_xn-x", "MI_xn-y", "MI_x-y"])
                        else:
                            plot_matrices([to_np_array(self.info_norm_all), to_np_array(self.negative_log_noise_amp_all.squeeze()), to_np_array(self.causality_pred_all.squeeze())], images_per_row = 4,
                                          subtitles = ["info-norm", "-log10(Eta)", "causality prediction"],
                                         )
                            if is_plot_MI:
                                plot_matrices([MI_xn_x, MI_xn_y, MI_x_y], images_per_row = 4, subtitles = ["MI_xn-x", "MI_xn-y", "MI_x_y"])
                    if record_mode > 0:
                        record_data(self.data_record[target_id], [to_np_array(self.noise_amp_all), self.info_norm_all, i], ["noise_amp", "info_norm_all", "iter_interval"])


                    if A_whole is not None:
                        try:
                            if record_mode > 0:
                                record_data(self.data_record[target_id], [precision[1], recall[1], F1[1], ROC_AUC, PR_AUC], ["precision", "recall", "F1", "ROC_AUC", "PR_AUC"])
                            print("Causality prediction:\nprecision: {0:.9f}\trecall: {1:.9f}\tF1 : {2:.9f}\tROC_AUC: {3:.9f}\tPR_AUC: {4:.9f}".format(precision[1], recall[1], F1[1], ROC_AUC, PR_AUC))
                        except:
                            pass
                    print("\n" + "=" * 80 + "\n")

                # Saving:
                if i % save_interval == 0:
                    self.info_dict["data_record"] = self.data_record
                    self.info_dict["noise_amp_all"] = self.noise_amp_all
                    self.info_dict["info_norm_all"] = self.info_norm_all
                    self.info_dict["causality_pred_all"] = self.causality_pred_all
                    self.info_dict["negative_log_noise_amp_all"] = self.negative_log_noise_amp_all
                    if is_plot_MI:
                        self.info_dict["MI_xn_x"] = MI_xn_x
                        self.info_dict["MI_xn_y"] = MI_xn_y
                        self.info_dict["MI_x_y"] = MI_x_y

                if to_stop:
                    print("Early stopping at iter {0}".format(i))
                    break
            self.info_dict["model_full_{0}".format(target_id)] = self.model.model_dict
        return to_np_array(self.info_norm_all), self.info_dict


def plot_comparison(matrix, A_whole, title, isplot = True):
    if isplot:
        if A_whole is not None:
            causality_truth = np.abs(A_whole) > 0
            if noise_type == "uniform-series":
                causality_truth = causality_truth.any(-2, keepdims = True)
            plot_matrices([matrix, causality_truth.squeeze()], subtitles = [title, "Truth"], images_per_row = 4)
        else:
            plot_matrices([matrix], subtitles = [title])


# ## Causal learning with chosen method:

# In[ ]:


"""
The output is saved in the item_dict["result"], where item_dict["result"][0] is the causal matrix, where the (i, j)
    element is the inferred causal strength from j to i.
The item_dict["metrics"] records the AUC-PR and AUC-ROC metrics, compared with the true causal matrix A_whole. A_whole has 
    dimension of (num_time_series, K, num_time_series), where num_time_series = N // group_sizes, with its (i, k, j) element
    indicating the true causal strength from j to i at a lag of k.
"""

learned_dict = {}
item_dict = {}
Ny_size = y.shape[-1] if X.shape[-1] != y.shape[-1] else group_sizes
struct_param = [[num_neurons, "Simple_Layer", {}] for num_neurons in struct_tuple] + [[(K2, Ny_size), "Simple_Layer", {"activation": "linear"}]]
settings = {"activation": activation}
if method[0] == "MI":
    """Mutual information method"""
    learned_dict[method[0]] = item_dict
    neighbors = method[1]
    item_dict["result"] = get_mutual_information(X_train, y_train, 
                                                 group_sizes = group_sizes, 
                                                 neighbors = neighbors, 
                                                 assigned_target_id = assigned_target_id, 
                                                 isplot = isplot,
                                                )
    plot_comparison(item_dict["result"][0], A_whole, "mutual_information", isplot = isplot)
    item_dict["metrics"] = get_AUCs(item_dict["result"][0], A_whole, neglect_idx = neglect_idx)
    pickle.dump(learned_dict, open(filename, "wb"))


elif method[0] == "trans-entropy":
    """Transfer entropy method"""
    learned_dict[method[0]] = item_dict
    neighbors = method[1]
    item_dict["result"] = get_conditional_MI(X_train, y_train, group_sizes = group_sizes, neighbors = neighbors, assigned_target_id = assigned_target_id)
    plot_comparison(item_dict["result"][0], A_whole, "conditional_MI", isplot = isplot)
    item_dict["metrics"] = get_AUCs(item_dict["result"][0], A_whole, neglect_idx = neglect_idx)
    pickle.dump(learned_dict, open(filename, "wb"))


elif method[0] == "G-linear":
    """Linear Granger method"""
    fit_intercept = method[1]
    learned_dict[method[0]] = item_dict
    item_dict["result"] = get_Granger_linear_reg(X_train, y_train, group_sizes = group_sizes, assigned_target_id = assigned_target_id, fit_intercept = fit_intercept)
    plot_comparison(item_dict["result"][0], A_whole, "Granger_causality", isplot = isplot)
    try:
        item_dict["metrics"] = get_AUCs(item_dict["result"][0], A_whole, neglect_idx = neglect_idx)
    except:
        item_dict["metrics"] = None
    pickle.dump(learned_dict, open(filename, "wb"))
    

elif method[0] == "elasticNet":
    """Elastic Net method"""
    learned_dict[method[0]] = item_dict
    item_dict["result"] = elastic_Net(X_train, y_train,
                                      group_sizes = group_sizes,
                                      assigned_target_id = assigned_target_id,
                                     )
    plot_comparison(item_dict["result"][0], A_whole, "elasticNet", isplot = isplot)
    item_dict["metrics"] = get_AUCs(item_dict["result"][0], A_whole, neglect_idx = neglect_idx)
    pickle.dump(learned_dict, open(filename, "wb"))


elif method[0] == "causal-influence":
    """Causal Influence method:"""
    learned_dict[method[0]] = item_dict
    reg_dict = {"weight": 1e-7, "bias": 1e-7}
    average_times = 5
    noise_amp = method[1]
    if noise_amp is not None:
        noise_amp = torch.ones(N).to(device) * noise_amp
        X_std = X_train.view(-1, N).std(0)
        noise_amp = noise_amp * X_std
    item_dict["result"] = get_causal_influence(X_train, y_train,
                                              validation_data = (X_test, y_test),
                                              group_sizes = group_sizes,
                                              struct_param = struct_param,
                                              average_times = average_times,
                                              causality_truth = causality_truth,
                                              noise_amp = noise_amp,
                                              settings = settings,
                                              is_cuda = is_cuda,
                                              patience = None,
                                              reg_dict = reg_dict,
                                              isplot = isplot,
                                             )
    item_dict["metrics"] = get_AUCs(item_dict["result"][0], A_whole, neglect_idx = neglect_idx)
    pickle.dump(learned_dict, open(filename, "wb"))

elif method[0] == "MPIR":
    """Our MPIR method"""
    noise_type = method[1]
    added_noise_type = method[2]
    noise_loss_scale = method[3]
    norm_mode = method[4]
    info_estimate_mode = method[5]
    warmup = 400
    max_iter = 30000
    learned_dict[method[0]] = item_dict
    record_mode = 1 if isplot else 0
    patience = None
    mpir = MPIR()
    item_dict["result"] = mpir.train(
        X = X_train,
        y = y_train,
        validation_data = (X_test, y_test),
        group_sizes = group_sizes,
        noise_type = noise_type,
        noise_loss_scale = noise_loss_scale,
        norm_mode = norm_mode,
        info_estimate_mode = info_estimate_mode,
        added_noise_type = added_noise_type,
        struct_param = struct_param,
        settings = settings,
        A_whole = A_whole,
        assigned_target_id = assigned_target_id,
        loss_core = loss_core,
        lr = lr,
        reg_amp = reg_amp,
        batch_size = batch_size,
        warmup = warmup,
        max_iter = max_iter,
        patience = patience,
        record_mode = record_mode,
        isplot = isplot,
        is_plot_MI = False,
        inspect_interval = 1 if info_estimate_mode == "var" else 20,
        plot_interval = 5 if info_estimate_mode == "var" else 200
    )
    item_dict["metrics"] = get_AUCs(item_dict["result"][0], A_whole, neglect_idx = neglect_idx)
    pickle.dump(learned_dict, open(filename, "wb"))

else:
    raise Exception("Method {0} not valid!".format(method[0]))

