# need implement the stochastic gradient descent from this
# https://realpython.com/gradient-descent-algorithm-python/#stochastic-gradient-descent-algorithms
import pickle  # thư viện dùng để lưu lại model

import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

with open('../data/diabetes_train.pkl', 'rb') as f:
    diabetes_train = pickle.load(f)

X = diabetes_train['data']
Y = diabetes_train['target']

print("Số chiều input: ", diabetes_train['data'].shape)
print("Số chiều target y tương ứng: ", diabetes_train['target'].shape)
print()

print("2 mẫu dữ liệu đầu tiên:")
print("input: ", diabetes_train['data'][:2])
print("target: ", diabetes_train['target'][:2])


# sử dụng sklearn để so sánh kết quả ( sklearn dùng thuật toán OLS )
def sklearn_linear_regression(_dataset):
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(_dataset['data'], _dataset['target'])

    return np.append(linear_regression.intercept_, linear_regression.coef_)


def gradient(_w):
    _N = diabetes_train['data'].shape[0]
    return (1 / _N) * X.T.dot(X.dot(_w) - Y)


def gradient_i(_w, _i):
    return X[_i].T.dot(X[_i].dot(_w) - Y[_i])


def mini_dataset_gradient(_w, _mini_dataset):
    _N = _mini_dataset.shape[0]
    _X = _mini_dataset[:, :-1]
    _one_vector = np.ones((_mini_dataset.shape[0], 1), dtype=_mini_dataset.dtype)
    _X = np.hstack([_one_vector, _X])
    _Y = _mini_dataset[:, -1]
    return (1 / _N) * _X.T.dot(_X.dot(_w) - _Y)


def cost(_w):
    _N = diabetes_train['data'].shape[0]
    return .5 / _N * np.linalg.norm(Y - X.dot(_w))


def gradient_descent(_w_init, eta):
    i = 0
    _w = _w_init.copy()
    while True:
        grad = gradient(_w)
        _w = _w - eta * grad
        if np.linalg.norm(gradient(_w)) / len(_w) < 1e-3:
            break
        i += 1

    return _w, i


# Stochastic Gradient Descent
def stochastic_gradient_descent(_w_init, eta):
    iter_check = 10
    _N = X.shape[0]
    _w = [_w_init]
    _w_last_check = _w_init.copy()
    count = 0
    for _epoch in range(50):
        random_id = np.random.permutation(_N)
        for _i in range(_N):
            count += 1
            _gradient = gradient_i(_w[-1], random_id[_i])
            _w_new = _w[-1] - eta * _gradient
            _w.append(_w_new)
            if count % iter_check == 0:
                _current_w = _w[-1]
                if np.linalg.norm(_current_w - _w_last_check) / len(_current_w) < 1e-3:
                    return _w, _epoch, count
                _w_last_check = _current_w

    return _w, _epoch, count


# Mini-batch Gradient Descent
def mini_batch_gradient_descent(_w_init, _epoches=50, _batch_size=50, _eta=.1):
    _dataset = np.hstack([diabetes_train['data'], diabetes_train['target'].reshape(-1, 1)])
    _N = _dataset.shape[0]
    _w = [_w_init]
    _v_old = np.zeros_like(_w_init)
    _gamma = .4
    _global_minimum = _w_init.copy()

    for _epoch in range(_epoches):
        np.random.shuffle(_dataset)
        for start in range(0, _N, _batch_size):
            end = start + _batch_size
            _mini_dataset = _dataset[start:end, :]
            _gradient = mini_dataset_gradient(_w[-1], _mini_dataset)
            _v_new = _gamma * _v_old + _eta * _gradient
            # _delta_w = _eta * _gradient
            # _w_new = _w[-1] - _delta_w
            _global_minimum = _global_minimum - _v_new
            _w.append(_global_minimum)
            if np.linalg.norm(_gradient) < 1e-3:
                return _w, _epoch
            _v_old = _v_new
    return _w, _epoch
    # Run program and compare with sklearn


def visualize(_w):
    _x = []
    _y = []
    if hasattr(_w, 'shape') and len(_w.shape) == 1:
        _x.append(np.linalg.norm(_w))
        _y.append(cost(_w))
        return _x, _y
    else:
        for _i in range(len(_w)):
            _x.append(np.linalg.norm(_w[_i]))
            _y.append(cost(_w[_i]))
        return _x, _y


if __name__ == '__main__':
    # init algorithm
    w_init = np.random.rand(diabetes_train['data'].shape[1] + 1) * 100
    one_vector = np.ones((X.shape[0], 1), dtype=X.dtype)
    X = np.hstack([one_vector, X])

    # sklearn (ols)
    w_sklearn = sklearn_linear_regression(diabetes_train)
    print('cost(w_sklearn) = ', cost(w_sklearn))

    # using gradient descent (batch)
    # w_gd, iterations = gradient_descent(w_init, .1)

    # using stochastic gradient descent
    w_scd, _, _ = stochastic_gradient_descent(w_init, .8)
    print('cost(w_scd) = ', cost(w_scd[-1]))

    # using mini-batch gradient descent
    w_mgcd, _ = mini_batch_gradient_descent(w_init, 50, 80, .4)
    print('cost(w_mgcd) = ', cost(w_mgcd[-1]))

    # VISUALIZE
    fix, axs = plt.subplots(2, 2, constrained_layout=True)
    # fix.tight_layout()
    plt.figure(figsize=(10, 8))
    # SKLEARN
    x, y = visualize(w_sklearn)
    axs[0, 0].plot(x, y, 'o')
    axs[0, 0].set_xlabel('norm w')
    axs[0, 0].set_ylabel('cost')
    axs[0, 0].set_title('sklearn accurate')

    # Stochastic GD
    x, y = visualize(w_scd)
    axs[0, 1].plot(x, y)
    axs[0, 1].set_xlabel('norm w')
    axs[0, 1].set_ylabel('cost')
    axs[0, 1].set_title('Stochastic GD')

    # mini-batch GD
    x, y = visualize(w_mgcd)
    axs[1, 0].plot(x, y)
    axs[1, 0].set_xlabel('norm w')
    axs[1, 0].set_ylabel('cost')
    axs[1, 0].set_title('Mini-batch GD')
    plt.show()
