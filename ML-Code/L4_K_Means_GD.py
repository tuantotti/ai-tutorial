import numpy as np
import pandas as pd

# Define constant
blobs = pd.read_csv('./data/kmeans_blobs.csv')
dataset = blobs.iloc[:, 1:3]
# print(blobs)
k = 3
options = 2
forget_rate = 0.8


# 0: Standard kmeans
# 1: Stochastic Gradient Descent kmeans
# 2: Mini-batch Stochastic Gradient Descent kmeans


def kmeans(_k, _dataset, _mini_batch_size=5):
    _centroids = _dataset.sample(_k)
    _iterations = _dataset.shape[0]
    _cluster_indexes = np.array([])

    for i in range(_iterations):
        x_m = _dataset.sample(_mini_batch_size)
        for j in range(x_m.shape[0]):
            _x = x_m.iloc[j, :]
            _cluster_index, _error = nearest_cluster(_x, _centroids)
            w_k = _centroids.iloc[_cluster_index, :]
            _centroids.iloc[_cluster_index, :] = w_k + learning_rate(i, 2, forget_rate) * (_x - w_k)

    return _centroids


def rss_err(_arr_1, _arr_2):
    return np.square(np.sum((_arr_1 - _arr_2) ** 2))  # square to avoid the data point that it is much close


def learning_rate(_t, _n, _k):
    """
    _t is the t th iteration
    _k, _n are positive constants
    _k exists in (0.5,1] is called forgetting rate
    """
    return (_t + _n) ** _k


def nearest_cluster(_x, _centroids):
    _n = _centroids.shape[0]
    _all_errors = []
    for i in range(_n):
        err = rss_err(_x, _centroids.iloc[i, :])
        _all_errors.append(err)

    _nearest_cluster = np.where(_all_errors == np.amin(_all_errors))[0]  # return array of minimum errors
    _cluster_index = _nearest_cluster[0]
    return _cluster_index, _all_errors[_cluster_index]


if options == 2:
    kmeans(k, dataset)
