import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

blobs = pd.read_csv('./data/kmeans_blobs.csv')  # return DataFrame


def initiate_centroids(_k, _dset):
    return _dset.sample(_k)  # choose _k random sample


def rss_err(_a, _b):
    return np.square(np.sum((_a - _b) ** 2))


def assign_centroid(_dset, _centroids):
    """
    find the nearest from obs to k th centroid
    """
    _k = _centroids.shape[0]
    _n = _dset.shape[0]
    assignation = []  # assignation[i] is the value of centroid that the i th obs belongs to
    assign_errors = []
    for obs in range(_n):
        all_errors = np.array([])
        for centroid in range(_k):
            err = rss_err(_centroids.iloc[centroid, :], _dset.iloc[obs, :])
            all_errors = np.append(all_errors, err)
        min_indexes = np.where(all_errors == np.amin(all_errors))  # return a boolean ndarray
        nearest_centroid = min_indexes[0].tolist()[0]
        nearest_centroid_error = np.amin(all_errors)

        assignation.append(nearest_centroid)
        assign_errors.append(nearest_centroid_error)

    return assignation, assign_errors


def kmeans(_dset, _k=0, _tol=1e-4):
    dset_cpy = _dset.copy()
    err = []
    is_convergence = False
    j = 0

    _centroids = initiate_centroids(_k, _dset)

    while not is_convergence:
        # for every iteration, it needs to use the whole dataset and calculate the nearest data to centroid
        # --> optimize by use GD
        dset_cpy.loc[:, 'centroid'], j_err = assign_centroid(dset_cpy, _centroids)
        err.append(sum(j_err))

        _centroids = dset_cpy.groupby('centroid').agg('mean').reset_index(drop=True)  # recompute the centroids
        if j > 0:
            if err[j - 1] - err[j] <= _tol:
                is_convergence = True
        j += 1

    dset_cpy.loc[:, 'centroid'], j_err = assign_centroid(dset_cpy, _centroids)
    _centroids = dset_cpy.groupby('centroid').agg('mean').reset_index(drop=True)

    return dset_cpy['centroid'], j_err, _centroids


np.random.seed(42)
df = blobs[['x', 'y']]
df.loc[:, 'centroid'], df.loc[:, 'error'], centroids = kmeans(df[['x', 'y']], 3)
df.head()
print(centroids)

ig, ax = plt.subplots(figsize=(8, 6))
colorMap = ListedColormap(["crimson", "mediumblue", "darkmagenta"])
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], marker='o', c=df['centroid'].astype('category'), cmap=colorMap, s=80,
            alpha=0.5)
plt.scatter(centroids.iloc[:, 0], centroids.iloc[:, 1], marker='s', s=200, c=[0, 1, 2], cmap=colorMap)
ax.set_xlabel(r'x', fontsize=14)
ax.set_ylabel(r'y', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
