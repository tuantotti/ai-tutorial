import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse


def softmax(_z):
    _p_z = np.exp(_z - np.max(_z, axis=0, keepdims=True))
    return _p_z / _p_z.sum(axis=0)


def convert_labels(_y, _C):
    _Y = sparse.coo_matrix((np.ones_like(_y), (_y, np.arange(len(_y)))), shape=(_C, len(_y))).toarray()
    return _Y


def softmax_regression(_X, _y, _w_init, eta, tol=1e-4, max_count=10000):
    _w = [_w_init]
    _C = _w_init.shape[1]
    _count = 0
    _epoch = 0
    _check = 20
    _N = _X.shape[1]
    _d = _X.shape[0]
    _Y = convert_labels(_y, _C)
    while _count < max_count:
        _epoch += 1
        _random_index = np.random.permutation(_N)
        for _i in _random_index:
            _xi = _X[:, _i].reshape(_d, 1)
            _yi = _Y[:, _i].reshape(_C, 1)
            _z = np.dot(_w[-1].T, _xi)
            _ai = softmax(_z)
            _w_new = _w[-1] + eta * _xi.dot((_yi - _ai).T)
            _count += 1
            if _count % _check == 0:
                if np.linalg.norm(_w_new - _w[-_check]) < tol:
                    return _w, _epoch
            _w.append(_w_new)

    return _w, _epoch


def pred(_w, _X):
    _A = softmax(_w.T.dot(_X))

    return np.argmax(_A, axis=0)


def display(_X, label):
    _X0 = _X[:, label == 0]
    _X1 = _X[:, label == 1]
    _X2 = _X[:, label == 2]

    plt.plot(_X0[0, :], _X0[1, :], 'b^', markersize=4, alpha=.8)
    plt.plot(_X1[0, :], _X1[1, :], 'go', markersize=4, alpha=.8)
    plt.plot(_X2[0, :], _X2[1, :], 'rs', markersize=4, alpha=.8)

    plt.show()


def show_result(_w, _X, _origin_label):
    xm = np.arange(-2, 11, 0.025)
    ym = np.arange(-3, 10, 0.025)
    xx, yy = np.meshgrid(xm, ym)

    xx1 = xx.ravel().reshape(1, xx.size)
    yy1 = yy.ravel().reshape(1, yy.size)

    # print(xx.shape, yy.shape)
    XX = np.concatenate((np.ones((1, xx.size)), xx1, yy1), axis=0)

    _Z = pred(w[-1], XX)

    # Put the result into a color plot
    _Z = _Z.reshape(xx.shape)

    plt.contourf(xx, yy, _Z, 200, cmap='jet', alpha=.1)

    plt.xlim(-2, 11)
    plt.ylim(-3, 10)
    plt.xticks(())
    plt.yticks(())
    display(_X[1:, :], _origin_label)
    plt.show()


if __name__ == '__main__':
    means = [[2, 2], [8, 3], [3, 6]]
    cov = [[1, 0], [0, 1]]
    N = 500
    X0 = np.random.multivariate_normal(means[0], cov, N)
    X1 = np.random.multivariate_normal(means[1], cov, N)
    X2 = np.random.multivariate_normal(means[2], cov, N)
    X = np.concatenate((X0, X1, X2), axis=0).T
    # extended data
    X = np.concatenate((np.ones((1, 3 * N)), X), axis=0)
    C = 3

    original_label = np.asarray([0] * N + [1] * N + [2] * N).T

    display(X[1:, :], original_label)
    w_init = np.random.randn(X.shape[0], C)
    w, epoch = softmax_regression(X, original_label, w_init, .05)
    print(w[-1])

    show_result(w, X, original_label)
