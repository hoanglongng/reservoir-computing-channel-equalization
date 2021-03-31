import numpy as np

def training(x, y):
    input = np.array([x]).T
    bias = np.ones(input.shape)
    X = np.concatenate((bias, input), axis = 1)
    Y = np.array([y]).T
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    return theta

def testing(x, theta):
    input = np.array([x]).T
    bias = np.ones(input.shape)
    X = np.concatenate((bias, input), axis = 1)
    Y_hat = X.dot(theta)

    return Y_hat
