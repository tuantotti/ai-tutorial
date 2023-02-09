import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def h(_x, _w):
    _value = np.dot(_w.T, _x)
    return np.sign(_value)


def has_converged(_X, _y, _w):
    return np.array_equal(h(_X, _w), _y)


def perceptron(_X, _y, _w_init):
    _w = [_w_init]
    _N = X.shape[1]
    _d = X.shape[0]
    mis_points = []
    while True:
        _random_index = np.random.permutation(_N)
        for i in range(_N):
            _xi = _X[:, _random_index[i]].reshape(_d, 1)
            _yi = _y[0, _random_index[i]]

            _value = h(_w[-1], _xi)
            if _value != _yi:
                mis_points.append(_random_index[i])
                _w_new = _w[-1] + _yi * _xi
                _w.append(_w_new)
        if has_converged(_X, _y, _w[-1]):
            break

    return _w, mis_points


def draw_line(_w):
    w0, w1, w2 = _w[0], _w[1], _w[2]
    if w2 != 0:
        x11, x12 = -100, 100
        return plt.plot([x11, x12], [-(w1 * x11 + w0) / w2, -(w1 * x12 + w0) / w2], 'k')
    else:
        x10 = -w0 / w1
        return plt.plot([x10, x10], [-100, 100], 'k')


def viz_alg_1d_2(_w):
    it = len(_w)
    fig, ax = plt.subplots(figsize=(5, 5))

    def update(i):
        ani = plt.cla()
        # points
        ani = plt.plot(X0[0, :], X0[1, :], 'b^', markersize=8, alpha=.8)
        ani = plt.plot(X1[0, :], X1[1, :], 'ro', markersize=8, alpha=.8)
        ani = plt.axis([0, 6, -2, 4])
        i2 = i if i < it else it - 1
        ani = draw_line(_w[i2])
        if i < it - 1:
            # draw one  misclassified point
            circle = plt.Circle((X[1, m[i]], X[2, m[i]]), 0.15, color='k', fill=False)
            ax.add_artist(circle)
        # hide axis
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_ticks([])
        cur_axes.axes.get_yaxis().set_ticks([])

        label = 'PLA: iter %d/%d' % (i2, it - 1)
        ax.set_xlabel(label)
        return ani, ax

    anim = FuncAnimation(fig, update, frames=np.arange(0, it + 2), interval=1000)
    # save
    anim.save('pla_vis.gif', dpi=100, writer='imagemagick')
    plt.show()


if __name__ == '__main__':
    np.random.seed(2)

    means = [[2, 2], [4, 2]]
    cov = [[.3, .2], [.2, .3]]
    N = 10
    X0 = np.random.multivariate_normal(means[0], cov, N).T
    X1 = np.random.multivariate_normal(means[1], cov, N).T

    X = np.concatenate((X0, X1), axis=1)
    y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis=1)
    X = np.concatenate((np.ones((1, 2 * N)), X), axis=0)
    d = X.shape[0]

    w_init = np.random.randn(d, 1)
    (w, m) = perceptron(X, y, w_init)
    viz_alg_1d_2(w)
    print()
