import numpy as np
from matplotlib import pyplot as plt
from numpy.core.shape_base import vstack
from numpy import linalg as lg

# m - количество примеров, n - количество признаков

# матричное произведение

def compute_hypothesis(X, theta):
    return X @ theta


def compute_cost(X, y, theta):
    m = X.shape[0]  # количество примеров в выборке
    # ВАШ КОД ЗДЕСЬ

    return 1 / (2 * m) * np.sum(np.square(compute_hypothesis(X, theta) - y))
    # ==============


# сравнение градиентного спуска с нормальным уравнением
def gradient_descend(X, y, theta, alpha, num_iter):
    history = list()
    m = X.shape[0]  # количество примеров в выборке
    n = X.shape[1]  # количество признаков с фиктивным
    for i in range(num_iter):

        theta_temp = theta

        for j in range(n):
            theta_temp[j] = theta_temp[j] - alpha * (compute_hypothesis(X, theta) - y).dot(X[:, j]) / m
            # ВАШ КОД ЗДЕСЬ
            # =====================

        theta = theta_temp
        history.append(compute_cost(X, y, theta))

    return history, theta


def scale_features(X):
    # ВАШ КОД ЗДЕСЬ
    X_temp = X[:, 1:]
    m = X_temp.shape[0]
    n = X_temp.shape[1]

    mu = [0] * n
    sigma = [0] * n
    res = []

    for j in range(n):
        mu[j] = 1 / m * sum(X_temp[:, j])
        sigma[j] = np.sqrt(1 / (m - 1) * sum((X_temp[:, j] - mu[j]) ** 2))
        res.append((X_temp[:, j] - mu[j]) / sigma[j])

    X = vstack((X[:, 0], np.array(res))).T
    return X
    # =====================


# точная оценка оптимальных параметров (тета)
def normal_equation(X, y):
    # ВАШ КОД ЗДЕСь
    return np.dot(np.dot(lg.inv(np.dot(X.T, X)), X.T), y)
    # X_T = np.transpose(X)
    # return np.matmul(np.matmul(lg.inv(np.matmul(X_T, X)), X_T), y)
    # return lg.pinv(X.T @ X) @ X.T @ y

    # =====================


def load_data(data_file_path):
    with open(data_file_path) as input_file:
        X = list()
        y = list()
        for line in input_file:
            *row, label = map(float, line.split(','))
            X.append([1] + row)
            y.append(label)
        return np.array(X, float), np.array(y, float)


X, y = load_data('lab1data2.txt')

history, theta = gradient_descend(X, y, np.array([0.9, 0., 0.4], float), 0.009, 1500)

plt.title('График изменения функции стоимости от номера итерации до стандартизации')
plt.plot(range(len(history)), history)
plt.show()

X = scale_features(X)

history, theta = gradient_descend(X, y, np.array([0.9, 0., 0.4], float), 0.009, 1500)

plt.title('График изменения функции стоимости от номера итерации после стандартизации')
plt.plot(range(len(history)), history)
plt.show()

theta_solution = normal_equation(X, y)
print(f'theta, посчитанные через градиентный спуск: {theta}, через нормальное уравнение: {theta_solution}')
