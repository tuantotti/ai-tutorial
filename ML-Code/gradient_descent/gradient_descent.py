import math

import numpy as np

learning_rate = .1


def f(x):
    return x ** 2 + 5 * math.sin(x)


def f_derivative(x):
    return 2 * x + 5 * math.cos(x)


def gradient_descent(_x0, _learning_rate):
    _x = [_x0]
    for i in range(100):
        x_new = _x[-1] - _learning_rate * f_derivative(_x[-1])
        """
        Khi giá trị đạo hàm tiến đến gần 0 --> local min --> dừng thuật toán
        """
        if abs(f_derivative(x_new)) <= 1e-3:
            break
        _x.append(x_new)

    return _x, i


def numerical_gradient_descent(_w):
    epochs = 1e-4
    _g = np.zeros_like(_w)
    for i in range(len(_w)):
        w_p = _w[i] + epochs
        w_n = _w[i] - epochs
        _g[i] = (f(w_p) - f(w_n)) / (2 * epochs)

    return _g


def check_gradient_descent(_w):
    grad1 = []
    if len(_w) > 1:
        for i in range(len(_w)):
            grad1.append(f_derivative(_w[i]))
    else:
        grad1 = f_derivative(_w)

    grad2 = numerical_gradient_descent(_w)

    # dùng hàm norm để tính chuẩn của matrix thường là căn của tổng bình phương tất cả thành phần
    # trong matrix
    return True if np.linalg.norm(grad1 - grad2) < 1e-6 else False


(x1, it1) = gradient_descent(-5, .1)
(x2, it2) = gradient_descent(5, .1)

print('Solution x1 = %f, cost = %f, after %d iterations' % (x1[-1], f(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, after %d iterations' % (x2[-1], f(x2[-1]), it2))
a = np.random.rand(3, 1)
print(check_gradient_descent(np.random.rand(3, 1)))
