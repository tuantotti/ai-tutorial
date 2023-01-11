import math

import numpy as np


def f(x):
    return x ** 2 + 5 * math.sin(x)


def gradient(x):
    return 2 * x + 5 * math.cos(x)


def has_convergence(_theta):
    return np.linalg.norm(gradient(_theta)) <= 1e-3


# batch gradient descent
def gradient_descent_momentum(_init_theta, _gamma, _eta):
    _global_minimum = _init_theta
    _v_old = np.zeros_like(_init_theta)
    for i in range(1000):
        _v_new = _gamma * _v_old + _eta * gradient(_global_minimum)
        _global_minimum = _global_minimum - _v_new
        if has_convergence(_global_minimum):
            break
        _v_old = _v_new
    return _global_minimum, i


# batch gradient descent
def nesterov_accelerated_gradient(_init_theta, _gamma, _eta):
    _global_minimum = _init_theta
    _v_old = np.zeros_like(_init_theta)
    for i in range(1000):
        _v_new = _gamma * _v_old + _eta * gradient(_global_minimum - _gamma * _v_old)
        _global_minimum = _global_minimum - _v_new
        if has_convergence(_global_minimum):
            break
        _v_old = _v_new
    return _global_minimum, i


gamma = .9
eta = .1
(x1, it1) = gradient_descent_momentum(-6, gamma, eta)
(x2, it2) = gradient_descent_momentum(6, gamma, eta)
print('Solution x1 = %f, cost = %f, after %d iterations' % (x1, f(x1), it1))
print('Solution x2 = %f, cost = %f, after %d iterations' % (x2, f(x2), it2))
(x3, it3) = nesterov_accelerated_gradient(-6, gamma, eta)
print('Solution x1 = %f, cost = %f, after %d iterations' % (x3, f(x3), it3))
