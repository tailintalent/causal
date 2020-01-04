import numpy as np
import matplotlib.pylab as plt
import torch
from copy import deepcopy

def plot_matrices(
    matrix_list, 
    shape = None, 
    images_per_row = 10, 
    scale_limit = None,
    figsize = (20, 8), 
    x_axis_list = None,
    filename = None,
    title = None,
    subtitles = [],
    highlight_bad_values = True,
    plt = None,
    pdf = None,
    ):
    """Plot the images for each matrix in the matrix_list."""
    import matplotlib
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize = figsize)
    fig.set_canvas(plt.gcf().canvas)
    if title is not None:
        fig.suptitle(title, fontsize = 18, horizontalalignment = 'left', x=0.1)
    
    num_matrixs = len(matrix_list)
    rows = np.ceil(num_matrixs / float(images_per_row))
    try:
        matrix_list_reshaped = np.reshape(np.array(matrix_list), (-1, shape[0],shape[1])) \
            if shape is not None else np.array(matrix_list)
    except:
        matrix_list_reshaped = matrix_list
    if scale_limit == "auto":
        scale_min = np.Inf
        scale_max = -np.Inf
        for matrix in matrix_list:
            scale_min = min(scale_min, np.min(matrix))
            scale_max = max(scale_max, np.max(matrix))
        scale_limit = (scale_min, scale_max)
    for i in range(len(matrix_list)):
        ax = fig.add_subplot(rows, images_per_row, i + 1)
        image = matrix_list_reshaped[i].astype(float)
        if len(image.shape) == 1:
            image = np.expand_dims(image, 1)
        if highlight_bad_values:
            cmap = matplotlib.cm.binary
            cmap.set_bad('red', alpha = 0.2)
            mask_key = []
            mask_key.append(np.isnan(image))
            mask_key.append(np.isinf(image))
            mask_key = np.any(np.array(mask_key), axis = 0)
            image = np.ma.array(image, mask = mask_key)
        else:
            cmap = matplotlib.cm.binary
        if scale_limit is None:
            ax.matshow(image, cmap = cmap)
        else:
            assert len(scale_limit) == 2, "scale_limit should be a 2-tuple!"
            ax.matshow(image, cmap = cmap, vmin = scale_limit[0], vmax = scale_limit[1])
        if len(subtitles) > 0:
            ax.set_title(subtitles[i])
        try:
            xlabel = "({0:.4f},{1:.4f})\nshape: ({2}, {3})".format(np.min(image), np.max(image), image.shape[0], image.shape[1])
            if x_axis_list is not None:
                xlabel += "\n{0}".format(x_axis_list[i])
            plt.xlabel(xlabel)
        except:
            pass
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)
    if pdf is not None:
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
    else:
        plt.show()

    if scale_limit is not None:
        print("scale_limit: ({0:.6f}, {1:.6f})".format(scale_limit[0], scale_limit[1]))
    print()


class Early_Stopping(object):
    def __init__(self, patience = 100, epsilon = 0, mode = "min"):
        self.patience = patience
        self.epsilon = epsilon
        self.mode = "min"
        self.best_value = None
        self.wait = 0
        
    def monitor(self, value):
        to_stop = False
        if self.patience is not None:
            if self.best_value is None:
                self.best_value = value
                self.wait = 0
            else:
                if (self.mode == "min" and value < self.best_value - self.epsilon) or \
                   (self.mode == "max" and value > self.best_value + self.epsilon):
                    self.best_value = value
                    self.wait = 0
                else:
                    if self.wait >= self.patience:
                        to_stop = True
                    else:
                        self.wait += 1
        return to_stop


def record_data(data_record_dict, data_list, key_list):
    """Record data to the dictionary data_record_dict. It records each key: value pair in the corresponding location of 
    key_list and data_list into the dictionary."""
    assert len(data_list) == len(key_list), "the data_list and key_list should have the same length!"
    for data, key in zip(data_list, key_list):
        if key not in data_record_dict:
            data_record_dict[key] = [data]
        else: 
            data_record_dict[key].append(data)


def make_dir(filename):
    import os
    import errno
    if not os.path.exists(os.path.dirname(filename)):
        print("directory {0} does not exist, created.".format(os.path.dirname(filename)))
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                print(exc)
            raise


def norm(matrix, noise_mode, mode = "L1"):
    # Clamp the matrix if noise_mode is "permutation":
    if noise_mode == "permutation":
        matrix = matrix.clamp(1e-6, 1)

    if mode == "L1":
        return matrix.abs().mean()
    elif mode == "L2":
        return torch.sqrt((matrix ** 2).mean() + 1e-9)
    elif mode == "info":
        return torch.log2(1 + 1 / matrix ** 2).sum() / 2
    elif isinstance(mode, tuple):
        if mode[0] == "order":
            order = mode[1]
            return (matrix.abs() ** order).mean()
        elif mode[0] == "exp_order":
            order = mode[1]
            return (torch.exp(matrix.abs() ** order)).mean()
        elif mode[0] == "log_order":
            order = mode[1]
            return (torch.log(matrix.abs() ** order + 1e-9)).mean()
        else:
            raise
    else:
        raise Exception("mode not recognized!".format(mode))


def train_test_split(X, y, test_size = 0.1):
    import torch
    if len(X.shape) == 4:
        X = X.view(-1, *X.shape[2:])
        y = y.view(-1, *y.shape[2:])
    num_examples = len(X)
    if test_size is not None:
        num_test = int(num_examples * test_size)
        num_train = num_examples - num_test
        idx_train = np.random.choice(range(num_examples), size = num_train, replace = False)
        idx_test = set(range(num_examples)) - set(idx_train)
        device = torch.device("cuda" if X.is_cuda else "cpu")
        idx_train = torch.LongTensor(list(idx_train)).to(device)
        idx_test = torch.LongTensor(list(idx_test)).to(device)
        X_train = X[idx_train]
        y_train = y[idx_train]
        X_test = X[idx_test]
        y_test = y[idx_test]
    else:
        X_train, X_test = X, X
        y_train, y_test = y, y
    return (X_train, y_train), (X_test, y_test)


def new_dict(Dict, new_content_dict):
    from copy import deepcopy
    new_Dict = deepcopy(Dict)
    new_Dict.update(new_content_dict)
    return new_Dict

def format_list(List, interval = "\t", decimals = None):
    if decimals is None:
        return interval.join(["{0}".format(element) for element in List])
    else:
        return interval.join(["{0:.{1}f}".format(element, decimals) for element in List])
    

def sort_two_lists(list1, list2, reverse = False):
    from operator import itemgetter
    if reverse:
        List = deepcopy([list(x) for x in zip(*sorted(zip(deepcopy(list1), deepcopy(list2)), key=itemgetter(0), reverse=True))])
    else:
        List = deepcopy([list(x) for x in zip(*sorted(zip(deepcopy(list1), deepcopy(list2)), key=itemgetter(0)))])
    if len(List) == 0:
        return [], []
    else:
        return List[0], List[1]


def get_args(arg, arg_id = 1, type = "str"):
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        arg_return = arg
    except:
        import sys
        try:
            arg_return = sys.argv[arg_id]
            if type == "int":
                arg_return = int(arg_return)
            elif type == "float":
                arg_return = float(arg_return)
            elif type == "bool":
                arg_return = eval(arg_return)
            elif type == "eval":
                arg_return = eval(arg_return)
            elif type == "tuple":
                if arg_return[0] not in ["(", "["]:
                    arg_return = eval(arg_return)
                else:
                    splitted = arg_return[1:-1].split(",")
                    List = []
                    for item in splitted:
                        try:
                            item = eval(item)
                        except:
                            pass
                        List.append(item)
                    arg_return = tuple(List)
            elif type == "str":
                pass
            else:
                raise Exception("type {0} not recognized!".format(type))
        except:
#            raise
             arg_return = arg
    return arg_return


def normalize_tensor(X, y, normalize):
    import torch
    if isinstance(X, np.ndarray):
        assert len(X.shape) >= 3
        assert len(y.shape) >= 3
        XY = np.concatenate([X, y], -2)
        if normalize == 0:
            pass
        elif normalize == 1:
            mean = XY.mean()
            std = XY.std()
            X = (X - mean) / std
            y = (y - mean) / std
        elif normalize == 2:
            shape = tuple(XY.shape)
            N = shape[-1]
            XY_reshape = XY.reshape(-1, N)
            if len(shape) == 3:
                mean = XY_reshape.mean(0).reshape(1, 1, N)
                std = XY_reshape.std(0).reshape(1, 1, N)
            elif len(shape) == 4:
                mean = XY_reshape.mean(0).reshape(1, 1, 1, N)
                std = XY_reshape.std(0).reshape(1, 1, 1, N)
            else:
                raise
            X = (X - mean) / std
            y = (y - mean) / std
        elif normalize == 3:
            N = XY.shape[-1]
            X_new = []
            Y_new = []
            for i in range(N):
                xy_ele = XY[...,i]
                x_ele = X[...,i]
                y_ele = y[...,i]
                xy_max = xy_ele.max()
                xy_min = xy_ele.min()
                x_new = (x_ele - xy_min) / (xy_max -xy_min)
                y_new = (y_ele - xy_min) / (xy_max -xy_min)
                X_new.append(x_new)
                Y_new.append(y_new)
            X = np.stack(X_new, -1)
            y = np.stack(Y_new, -1)
        elif normalize == 4:
            # Make each (...,K,N) zero mean:
            X_shape = X.shape
            X = X.reshape(-1, *X_shape[-2:])
            X = X - X.mean(0, keepdims = True)
            X = X.reshape(*X_shape)
            
            # Make each (...,K,N) zero mean and unit std:
            y_shape = y.shape
            y = y.reshape(-1, *y_shape[-2:])
            y = (y - y.mean(0, keepdims = True)) / y.std(0, keepdims = True)
            y = y.reshape(*y_shape)
        else:
            raise
    else:
        assert len(X.size()) >= 3
        assert len(y.size()) >= 3
        XY = torch.cat([X, y], -2)
        if normalize == 0:
            pass
        elif normalize == 1:
            mean = XY.mean()
            std = XY.std()
            X = (X - mean) / std
            y = (y - mean) / std
        elif normalize == 2:
            shape = tuple(XY.shape)
            N = shape[-1]
            XY_reshape = XY.reshape(-1, N)
            if len(shape) == 3:
                mean = XY_reshape.mean(0).view(1, 1, N)
                std = XY_reshape.std(0).view(1, 1, N)
            elif len(shape) == 4:
                mean = XY_reshape.mean(0).view(1, 1, 1, N)
                std = XY_reshape.std(0).view(1, 1, 1, N)
            else:
                raise
            X = (X - mean) / (std + 1e-9)
            y = (y - mean) / (std + 1e-9)
        elif normalize == 3:
            N = XY.shape[-1]
            X_new = []
            Y_new = []
            for i in range(N):
                xy_ele = XY[...,i]
                x_ele = X[...,i]
                y_ele = y[...,i]
                xy_max = xy_ele.max()
                xy_min = xy_ele.min()
                x_new = (x_ele - xy_min) / (xy_max -xy_min)
                y_new = (y_ele - xy_min) / (xy_max -xy_min)
                X_new.append(x_new)
                Y_new.append(y_new)
            X = torch.stack(X_new, -1)
            y = torch.stack(Y_new, -1)
        elif normalize == 4:
            pass
#             # Make each (...,K,N) zero mean:
#             X_shape = X.shape
#             X = X.view(-1, *X_shape[-2:])
#             X = X - X.mean(0, keepdim = True)
#             X = X.view(*X_shape)
            
#             # Make each (...,K,N) zero mean and unit std:
#             y_shape = y.shape
#             y = y.view(-1, *y_shape[-2:])
#             y = (y - y.mean(0, keepdim = True)) / y.std(0, keepdim = True)
#             y = y.view(*y_shape)
        else:
            raise
    return X, y
