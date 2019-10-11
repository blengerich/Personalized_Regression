import numpy as np
from matplotlib import pyplot as plt
import time
from numba import jit


def load_delta_U(fname):
    # Assumes npz
    npz_arr = np.load(fname)
    delta_U = npz_arr['arr_0']
    print("Successfully Loaded covariate distances from {}".format(fname))
    return delta_U


def create_delta_U(dml, U, dU, K, N, fname, normalize=True):
    # Assumes fname is .npz
    print("Calculating Pairwise Co-Variate Distances...")
    t = time.time()
    delta_U = dml.make_covariate_distances(U, dU, K, N, normalize)
    #delta_U = dml.make_covariate_distance_function(U, dU, K)
    print("Finished. Took {:.3f} seconds.".format(time.time() - t))
    if fname is not None:
        print("Saving Pairwise Co-Variate Distances to {}".format(fname))
        np.savez_compressed(fname, delta_U)
    return delta_U

"""
# TODO: Deprecated?
def calc_prediction_error(Y, beta_hat, X, N):
    return 0.5*np.mean(np.square(Y - np.array([X[i].dot(beta_hat[i]) for i in range(N)])))

def calc_prediction_error_logistic(Y, beta_hat, X):
    return 0.5*np.mean([
        np.log(np.exp(X[i].dot(beta_hat[i])) + 1) - Y[i]*X[i].dot(beta_hat[i])
        for i in range(len(X))])
"""

def print_errors(calc_pred_err, estimations, err_name, fname="results.txt"):
    with open(fname, 'a') as out_file:
        print("="*20 + " {} Error ".format(err_name) + "="*20)
        print("="*20 + " {} Error ".format(err_name) + "="*20, file=out_file)
        pred_errs = []
        for (beta_hat, estimator_name) in estimations:
            err = calc_pred_err(beta_hat)
            pred_errs.append((err, estimator_name))
            print("{}:{:.4f}".format(estimator_name, err))
            print("{}:{:.4f}".format(estimator_name, err), file=out_file)
    return pred_errs


def plot_learned_betas(true_beta, estimations, U):
    fig = plt.figure()

    # Assumes the first value in each row of U is a category
    colors = ['blue', 'green', 'cyan', 'orange', 'red']
    true_color  = 'black'
    true_marker = '*'
    markers = ['+', 'o', '.', 'x', 'v']

    labels = set(U[:, 0])
    for i, label in enumerate(labels):
        ax = fig.add_subplot(len(labels)/2+1, 2, i+1)
        ax.set_title("Type={}".format(label))
        handles = []
        descriptions = []

        selection = U[:, 0] == label
        handle = ax.scatter(
            true_beta[selection, 0],
            true_beta[selection, 1],
            color=true_color, marker='*')
        handles.append(handle)
        descriptions.append('True Beta')
        for j, (estimation, estimator_name) in enumerate(estimations):
            #if 'Mixture' not in estimator_name:
            #   continue
            #print(estimation)
            handle = ax.scatter(
                estimation[selection, 0]+np.random.normal(0, .02, np.sum(selection)),
                estimation[selection, 1]+np.random.normal(0, .02, np.sum(selection)),
                color=colors[j], marker='+')
            handles.append(handle)
            descriptions.append(estimator_name)

    ax = fig.add_subplot(len(labels)/2+1, 2, i+2)
    plt.legend(handles, descriptions, loc='upper center', bbox_to_anchor=(0.5, 1.05),
        ncol=2, fancybox=True, shadow=True)

    plt.show()


#def softmax(x):
@jit(nopython=True)
def soft_normalize(x):
    """Compute softmax values for each sets of scores in x."""
    exps = np.exp(x)
    return exps / np.sum(exps)

"""
def soft_normalize(ar, thresh=1e-3):
    # Makes the values in the array sum to 1, and no value is smaller than thresh.
    ar = np.maximum(thresh, ar)
    ar /= np.sum(ar)
    return ar
"""

def float_or_zero(x):
    try:
        return float(x)
    except ValueError:
        return 0.


# TODO: Should do mean imputation, not 0.
def to_one_hot(U, should_change):
    if should_change[0]:
        one_hot = to_one_hot_one_feature(U[:, 0])
    else:
        one_hot = np.array([float_or_zero(U[i, 0]) for i in range(len(U))])
        one_hot = np.expand_dims(one_hot, 1)
    #print("One Hot First Feature Shape:{}".format(one_hot.shape))
    for j in range(1, U.shape[1]):
        if should_change[j]:
            #print("Changing {}".format(j))
            one_hot_feature = to_one_hot_one_feature(U[:, j])
            one_hot = np.hstack((one_hot, one_hot_feature))
        else:
            continuous_feature = np.array([float_or_zero(U[i, j]) for i in range(len(U))])
            continuous_feature = np.expand_dims(continuous_feature, 1)
            one_hot = np.hstack((one_hot, continuous_feature))
        #print(one_hot.shape)
    return one_hot


def to_one_hot_one_feature(U):
    """ Assumes U has a single feature.
    Returns matrix of size U.shape[0], number_unique + 1
    """
    as_set = set(U)
    set_as_list = list(as_set)
    one_hot = np.zeros((U.shape[0], len(as_set)))
    for i in range(U.shape[0]):
        one_hot[i, set_as_list.index(U[i])] = 1
    return one_hot

def to_color_map_one_feature(U):
    bad_vals = set(["None", None, "not reported"])
    as_set = set(U) - bad_vals
    if "Breast" in as_set:
        print("Using pre-defined list")
        set_as_list = ["Pancreas", "Skin", "Thyroid", "Prostate", "Eye", "Kidney", "Uterus", "Liver", "Bladder",
        "Colorectal", "Esophagus", "Head and Neck", "Lymph Nodes", "Bile Duct", "Stomach", "Breast", "Brain", "Lung", "Ovary"]
        set_as_list.reverse()
    else:
        set_as_list = list(as_set)
    one_hot = np.zeros((U.shape[0]))
    for i in range(U.shape[0]):
        try:
            one_hot[i] = set_as_list.index(U[i])
        except:
            one_hot[i] = -1
    return one_hot, set_as_list