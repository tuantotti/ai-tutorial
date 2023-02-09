import matplotlib.pyplot as plt
import numpy as np


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def logistic_regression(_w_init, _X, _y, _eta=.1, _max_count=10000):
    _w = [_w_init]
    _count = 0
    _N = _X.shape[1]
    _d = _X.shape[0]
    _check_after = 20
    while _count < _max_count:
        _random_index = np.random.permutation(_N)
        for _i in _random_index:
            _xi = X[:, _i].reshape(_d, 1)
            _yi = y[_i]
            # net input
            _zi = np.dot(_w[-1].T, _xi)
            # activation function
            _activation = sigmoid(_zi)

            _w_new = _w[-1] + _eta * (_yi - _activation) * _xi
            _count += 1
            if _count % _check_after == 0:
                if np.linalg.norm(_w_new - _w[-1]) < 1e-4:
                    return _w
            _w.append(_w_new)
    return _w


if __name__ == '__main__':
    np.random.seed(2)

    X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
                   2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
    y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

    X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
    w_init = np.random.randn(X.shape[0], 1)
    w = logistic_regression(w_init, X, y, 0.05)

    x0 = X[1, np.where(y == 0)][0]
    y0 = y[np.where(y == 0)]

    x1 = X[1, np.where(y == 1)][0]
    y1 = y[np.where(y == 1)]

    plt.plot(x0, y0, 'ro', markersize=8)
    plt.plot(x1, y1, 'bs', markersize=8)

    xx = np.linspace(0, 6, 2000)
    w0 = w[-1][0][0]
    w1 = w[-1][1][0]
    threshold = -w0 / w1
    yy = sigmoid(w0 + w1 * xx)
    plt.axis([-2, 8, -1, 2])
    plt.plot(xx, yy, 'g-', linewidth=2)
    plt.plot(threshold, .5, 'y^', markersize=8)
    plt.xlabel('studying hours')
    plt.ylabel('predicted probability of pass')
    plt.show()

    # prediction
    x_test = np.array([[1, 1], [6, 1]])
    y_pred = sigmoid(np.dot(w[-1].T, x_test))
    label = np.where(y_pred < 0, 0, 1)
    print(label)
