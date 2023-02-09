import numpy as np
from cffi.backend_ctypes import xrange
from matplotlib import pyplot as plt
from scipy import sparse


def init_data(_N, _d0, _C):
    _X = np.zeros((_d0, _N * _C))  # data matrix (each row = single example)
    _y = np.zeros(_N * _C, dtype='uint8')  # class labels

    for j in xrange(_C):
        ix = range(_N * j, _N * (j + 1))
        r = np.linspace(0.0, 1, _N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, _N) + np.random.randn(_N) * 0.2  # theta
        _X[:, ix] = np.c_[r * np.sin(t), r * np.cos(t)].T
        _y[ix] = j

    return _X, _y


def convert_labels(_y, _C):
    _Y = sparse.coo_matrix((np.ones_like(_y), (_y, np.arange(len(_y)))), shape=(_C, len(_y))).toarray()
    return _Y


def softmax(_z):
    _p_z = np.exp(_z - np.max(_z, axis=0, keepdims=True))
    return _p_z / _p_z.sum(axis=0)


def loss(_y, _output):
    return -np.sum(_y * np.log(_output)) / _y.shape[1]


def run(_X, _Y, _d0, _d1, _d2):
    _W1 = 0.01 * np.random.randn(_d0, _d1)
    _b1 = np.zeros((_d1, 1))
    _W2 = 0.01 * np.random.randn(_d1, _d2)
    _b2 = np.zeros((_d2, 1))
    _N = _X.shape[1]
    _eta = 1
    for i in xrange(10000):
        # Feedforward
        _Z1 = np.dot(_W1.T, _X) + _b1
        _A1 = np.maximum(_Z1, 0)  # ReLU activation
        _Z2 = np.dot(_W2.T, _A1) + _b2
        _output = softmax(_Z2)

        _loss = loss(_Y, _output)
        if i % 1000 == 0:
            print('loss = %f' % _loss)

        # Back-propagation
        _E2 = (_output - _Y) / _N
        _dW2 = np.dot(_A1, _E2.T)
        _db2 = np.sum(_E2, axis=1, keepdims=True)

        _E1 = np.dot(_W2, _E2)
        _E1[_Z1 <= 0] = 0  # ReLU gradient
        _dW1 = np.dot(_X, _E1.T)
        _db1 = np.sum(_E1, axis=1, keepdims=True)

        # Gradient Descent Update
        _W1 += -_eta * _dW1
        _b1 += -_eta * _db1
        _W2 += -_eta * _dW2
        _b2 += -_eta * _db2

    return _W1, _b1, _W2, _b2


def show_result(_X, _W1, _W2, acc):
    xm = np.arange(-1.5, 1.5, 0.025)
    xlen = len(xm)
    ym = np.arange(-1.5, 1.5, 0.025)
    ylen = len(ym)
    xx, yy = np.meshgrid(xm, ym)

    xx1 = xx.ravel().reshape(1, xx.size)
    yy1 = yy.ravel().reshape(1, yy.size)

    _X0 = np.vstack((xx1, yy1))

    _Z1 = np.dot(_W1.T, _X0) + b1
    _A1 = np.maximum(_Z1, 0)
    _Z2 = np.dot(_W2.T, _A1) + b2
    # predicted class
    _Z = np.argmax(_Z2, axis=0)

    _Z = _Z.reshape(xx.shape)
    CS = plt.contourf(xx, yy, _Z, 200, cmap='jet', alpha=.1)

    N = 100

    plt.plot(_X[0, :N], _X[1, :N], 'bs', markersize=7)
    plt.plot(_X[0, N:2 * N], _X[1, N:2 * N], 'g^', markersize=7)
    plt.plot(_X[0, 2 * N:], _X[1, 2 * N:], 'ro', markersize=7)
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_ticks([])
    cur_axes.axes.get_yaxis().set_ticks([])

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xticks(())
    plt.yticks(())
    plt.title('accuracy = %.2f %%' % acc)
    plt.show()


N = 100  # number of points per class
d0 = 2  # dimensionality
C = 3  # number of classes
X, y = init_data(N, d0, C)
Y = convert_labels(y, C)
d0 = 2
d1 = h = 200  # size of hidden layer
d2 = C = 3

W1, b1, W2, b2 = run(X, Y, d0, d1, d2)
Z1 = np.dot(W1.T, X) + b1
A1 = np.maximum(Z1, 0)
Z2 = np.dot(W2.T, A1) + b2
output = softmax(Z2)
predicted_class = np.argmax(output, axis=0)
accuracy = 100 * np.mean(predicted_class == y)
print('Training Accuracy = %f' % accuracy)
show_result(X, W1, W2, accuracy)
