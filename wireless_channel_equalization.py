"""
PERFORMANCE TEST IN SYMBOL ERROR RATE FOR WIRELESS COMMUNICATION CHANNEL.
Performance is evaluated with respect to signal-to-noise ratio (SNR)
"""
import numpy as np
import matplotlib.pyplot as plt
import channel.wireless as wl
import transceiver.receiver as rx
import neural_nets.linear_regression as lr
import neural_nets.esn_simplified as rc

# Select the equalizer to be used!
no_equalizer = 1
linear_equalizer = 1
RC_equalizer = 1

symbol_set = np.array([-3, -1, 1, 3])                                           # Set of available symbols to sample
num_bits = 10000                                                                 # No. of bits
num_train_bits = 6000
SNR_dB_set = np.arange(12, 33, 4)                                               # SNR range

# Loop over SNR range
if no_equalizer:
    SER_no_eq = np.zeros(len(SNR_dB_set))
if linear_equalizer:
    SER_LE = np.zeros(len(SNR_dB_set))
if RC_equalizer:
    SER_RC = np.zeros(len(SNR_dB_set))
index = 0

# Compute SER vs SNR_dB
for SNR_dB in SNR_dB_set:
    # Collect data for testing
    received_symbols_test, original_symbols_test = wl.data_generator(num_bits, symbol_set, SNR_dB)

    if no_equalizer:
        # Detect symbols
        detected_symbols_no_eq = rx.symbol_detector(received_symbols_test, original_symbols_test, num_bits, symbol_set)
        # SER calculation
        SER_no_eq[index] = rx.SER_calculation(detected_symbols_no_eq, original_symbols_test, num_bits)

    if linear_equalizer:
        # Collect data for training
        received_symbols_train, original_symbols_train = wl.data_generator(num_train_bits, symbol_set, SNR_dB)
        # Linear equalizer training
        theta0 = lr.training(received_symbols_train, original_symbols_train)
        # Linear equalizer testing
        Y_hat = lr.testing(received_symbols_test, theta0)
        # Detect symbols
        detected_symbols_LE = rx.symbol_detector(np.concatenate(Y_hat), original_symbols_test, num_bits, symbol_set)
        # SER calculation
        SER_LE[index] = rx.SER_calculation(detected_symbols_LE, original_symbols_test, num_bits)

    if RC_equalizer:
        # Collect data for training
        received_symbols_train, original_symbols_train = wl.data_generator(num_train_bits, symbol_set, SNR_dB)
        num_nodes = 50
        beta = 0.5
        alpha = 1.1
        act_func = "sigmoid"
        # RC equalizer training
        theta0, weights_samples = rc.training(received_symbols_train, original_symbols_train, num_nodes, alpha, beta, act_func)
        # RC equalizer testing
        Y_hat = rc.testing(received_symbols_test, theta0, weights_samples, num_nodes, alpha, beta, act_func)
        # Detect symbols
        detected_symbols_LE = rx.symbol_detector(Y_hat, original_symbols_test, num_bits, symbol_set)
        # SER calculation
        SER_RC[index] = rx.SER_calculation(detected_symbols_LE, original_symbols_test, num_bits)

    index += 1

# Plot
plt.figure(1)
plt.title("Symbol error rate in wireless communication channel")
if no_equalizer:
    plt.plot(SNR_dB_set, SER_no_eq)
if linear_equalizer:
    plt.plot(SNR_dB_set, SER_LE)
if RC_equalizer:
    plt.plot(SNR_dB_set, SER_RC)
plt.xlabel("SNR (dB)")
plt.xticks(SNR_dB_set)
plt.ylabel("SER")
plt.yscale("log")
plt.ylim([1e-3, 1e0])
plt.gca().legend(('No equalizer', 'Linear equalizer', 'RC equalizer'))
plt.show()
