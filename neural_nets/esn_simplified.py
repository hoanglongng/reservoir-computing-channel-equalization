import numpy as np
from neural_nets.activations import sigmoid

def training(received_symbols_train, original_symbols_train, num_nodes, alpha, beta, act_func, desync_num = 1):
    "Training"
    train_len = len(received_symbols_train)
    # Matrix initialization
    internal_states = np.zeros((num_nodes, train_len))
    internal_states_memory = np.zeros(num_nodes)
    input_reservoir = np.zeros((num_nodes, train_len))
    # Reservoir input preparation
    input_train = np.tile(received_symbols_train, (num_nodes, 1))                           # After hold operation
    weights_in_samples = np.random.rand(num_nodes, 1) - 0.5                                 # Random in [-1, 1]
    weights_in = np.tile(weights_in_samples, (1, train_len))                                # Matrix of weights

    input_reservoir = input_train * weights_in                                              # Weighted inputs to be fed to the reservoir

    # Training
    for n in range(train_len):
        if n >= 1:
            internal_states_memory = np.roll(internal_states[:, n - 1], desync_num)
            if n == 1:
                internal_states_memory[0] = 0
            else:
                internal_states_memory[0] = internal_states[-desync_num, n - 2]
        if act_func == "tanh":
            internal_states[:, n] = np.tanh(beta * input_reservoir[:, n] + alpha * internal_states_memory)
        elif act_func == "sigmoid":
            internal_states[:, n] = sigmoid(beta * input_reservoir[:, n] + alpha * internal_states_memory)
        elif act_func == "linear":
            internal_states[:, n] = beta * input_reservoir[:, n] + alpha * internal_states_memory

    # Normal equation
    X = internal_states.T
    Y = original_symbols_train.reshape((train_len, 1))
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

    return theta, weights_in_samples

def testing(received_symbols_test, theta, weights_in_samples, num_nodes, alpha, beta, act_func, desync_num = 1):
    "Testing"
    test_len = len(received_symbols_test)
    # Matrix initialization
    internal_states = np.zeros((num_nodes, test_len))
    internal_states_memory = np.zeros(num_nodes)
    input_reservoir = np.zeros((num_nodes, test_len))
    # Reservoir input preparation
    input_test = np.tile(received_symbols_test, (num_nodes, 1))                           # After hold operation
    weights_in = np.tile(weights_in_samples, (1, test_len))                                   # Matrix of weights

    input_reservoir = input_test * weights_in                                              # Weighted inputs to be fed to the reservoir

    # Training
    for n in range(test_len):
        if n >= 1:
            internal_states_memory = np.roll(internal_states[:, n - 1], desync_num)
            if n == 1:
                internal_states_memory[0] = 0
            else:
                internal_states_memory[0] = internal_states[-desync_num, n - 2]
        if act_func == "tanh":
            internal_states[:, n] = np.tanh(beta * input_reservoir[:, n] + alpha * internal_states_memory)
        elif act_func == "sigmoid":
            internal_states[:, n] = sigmoid(beta * input_reservoir[:, n] + alpha * internal_states_memory)
        elif act_func == "linear":
            internal_states[:, n] = beta * input_reservoir[:, n] + alpha * internal_states_memory

    # Predict
    X = internal_states.T
    Y_hat = X.dot(theta)
    Y_hat = Y_hat.reshape((test_len,))

    return Y_hat