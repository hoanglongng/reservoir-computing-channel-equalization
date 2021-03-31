import numpy as np
import matplotlib.pyplot as plt
from math import pi, log10, e
from activations import *

test_conventional_funcs = 0
test_optical_funcs = 1

"CONVENTIONAL ACTIVATION FUNCTIONS"
if test_conventional_funcs:
    x = np.linspace(-4, 4, num = 1000)

    # Tanh
    y_tanh = np.tanh(x)

    # Sigmoid
    y_sigmoid = sigmoid(x)

    # Sine
    y_relu = relu(x)

    # FESNA
    y_leaky_relu = leaky_relu(x)

    # Plot
    plt.figure(1)
    plt.title("Transfer functions of nonlinear activation functions")
    plt.plot(x, y_tanh, 'k')
    plt.plot(x, y_sigmoid, 'g')
    plt.plot(x, y_relu, 'b')
    plt.plot(x, y_leaky_relu, 'r')
    plt.gca().legend(('tanh', 'sigmoid', 'relu', 'leaky_relu'), loc = 'upper left')
    plt.xlabel("Input")
    plt.xlim(-4, 4)
    plt.xticks([-4, -2, 0, 2, 4])
    plt.ylim(-1.2, 4.2)
    plt.yticks([-1, 0, 1, 2, 3, 4])
    plt.ylabel("Output")
    plt.show()

"OPTICAL ACTIVATION FUNCTIONS"
if test_optical_funcs:
    I = np.array([160, 200, 240, 300, 340, 400, 520]) * 1e-3
    Pin = np.linspace(0, 0.5, 100) * 1e-3

    plt.figure(1)
    plt.title("Transfer function of SOA")

    Pout = np.zeros((len(I), len(Pin)))
    for i in range(len(I)):
        for j in range(len(Pin)):
            Pout[i, j] = SOA(Pin[j], I[i])
            j += 1
        plt.plot(Pin*1e3, Pout[i, :]*1e3)
        i += 1

    plt.xlabel('Pin (mW)')
    plt.ylabel('Pout (mW)')
    plt.gca().legend((I*1e3).astype(int), loc = 'upper right')
    plt.show()        