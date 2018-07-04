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


def create_delta_U(dmr, U, dU, K, N, fname, normalize=True):
    # Assumes fname is .npz
    print("Calculating Pairwise Co-Variate Distances...")
    t = time.time()
    delta_U = dmr.make_covariate_distances(U, dU, K, N, normalize)
    print("Finished. Took {:.3f} seconds.".format(time.time() - t))
    if fname is not None:
        print("Saving Pairwise Co-Variate Distances to {}".format(fname))
        np.savez_compressed(fname, delta_U)
    return delta_U


def print_errors(calc_pred_err, estimations, err_name):
    print("="*20 + " {} Error ".format(err_name) + "="*20)
    pred_errs = []
    for (beta_hat, estimator_name) in estimations:
        err = calc_pred_err(beta_hat)
        pred_errs.append((err, estimator_name))
        print("{}:{:.2f}".format(estimator_name, err))
    return pred_errs


@jit(nopython=True)
def soft_normalize(x):
    """Compute softmax values for each sets of scores in x."""
    exps = np.exp(x)
    return exps / np.sum(exps)


def float_or_zero(x):
    try:
        return float(x)
    except ValueError:
        return 0.


def to_one_hot(U, should_change):
    if should_change[0]:
        one_hot = to_one_hot_one_feature(U[:, 0])
    else:
        one_hot = np.array([float_or_zero(U[i, 0]) for i in range(len(U))])
        one_hot = np.expand_dims(one_hot, 1)
    for j in range(1, U.shape[1]):
        if should_change[j]:
            one_hot_feature = to_one_hot_one_feature(U[:, j])
            one_hot = np.hstack((one_hot, one_hot_feature))
        else:
            continuous_feature = np.array([float_or_zero(U[i, j]) for i in range(len(U))])
            continuous_feature = np.expand_dims(continuous_feature, 1)
            one_hot = np.hstack((one_hot, continuous_feature))
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
            handle = ax.scatter(
                estimation[selection, 0],
                estimation[selection, 1],
                color=colors[j], marker='+')
            handles.append(handle)
            descriptions.append(estimator_name)

    ax = fig.add_subplot(len(labels)/2+1, 2, i+2)
    plt.legend(handles, descriptions, loc='upper center', bbox_to_anchor=(0.5, 1.05),
        ncol=2, fancybox=True, shadow=True)

    plt.show()