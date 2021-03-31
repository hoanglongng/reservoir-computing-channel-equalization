import numpy as np
import matplotlib.pyplot as plt
import transceiver.transmitter as tx
import transceiver.receiver as rx
from neural_nets.esn import ESN

"PARAMETER DEFINITION"
M = 16                                                                  # Modulation order
num_symbols = 100000                                                     # No. of symbols
sps = 16                                                                # No. of samples/symbol
Pref_dBm = 10                                                           # Reference power level in dBm
Pref = 10 ** (Pref_dBm/10) * 1e-3                                       # Reference power level in W

"FIBER PARAMETERS"
lengthKm = 100                                                          # Fibre length in km
length = lengthKm * 1e3                                                 # Fibre length in m
loss = 0.2e-3                                                           # Fibre loss coefficient
dispersion = 0                                                          # Fibre dispersion coefficient
gamma = 1e-4                                                            # Fibre nonlinearity coefficient
Gain_dB  = loss*length                                                  # Gain in dB to compensate for loss over length
Gain     = 10**(Gain_dB/10)                                             # Gain in linear scale
NF_dB    = 4.5                                                          # Noise figure in dB
NF       = 10**(NF_dB/10)                                               # Noise figure in linear scale

"SIGNAL TRANSMITTER"
roll_off = 0.1
span = 1024
Ein, tx_symbols, Psig = tx.QAM(num_symbols, M, roll_off, span, sps, Pref)

"SIGNAL TRANSMISSION"
Eout = Ein * np.exp(1j * gamma * length * np.abs(Ein)**2)

"SIGNAL RECEIVER"
rx_symbols = rx.QAM(Eout, roll_off, span, sps, Psig, Pref)

"ESN EQUALIZER"
dataIn = tx_symbols
dataOut = rx_symbols
# ESN definition
inSize = 2
outSize = 2
resSize = 250
connectivity = 0.01              # IMPORTANT FACTOR
spectralRadius = 0.1             # IMPORTANT FACTOR
withFeedback = 0
inputScaling = 1
net = ESN(inSize, outSize, resSize, connectivity, spectralRadius, inputScaling, withFeedback)
# Training data division
initRunLen = 100
trainRunLen = 1000
testRunLen = 10000
ignoredLen = 0
# Training and testing data preparation
trainInput = np.c_[dataOut.real[ignoredLen:ignoredLen+initRunLen + trainRunLen], dataOut.imag[ignoredLen:ignoredLen+initRunLen + trainRunLen]]
trainOutput = np.c_[dataIn.real[ignoredLen:ignoredLen+initRunLen + trainRunLen], dataIn.imag[ignoredLen:ignoredLen+initRunLen + trainRunLen]]

testInput = np.c_[dataOut.real[initRunLen+trainRunLen:initRunLen+trainRunLen+testRunLen], dataOut.imag[initRunLen+trainRunLen:initRunLen + trainRunLen + testRunLen]]
testOutput = np.c_[dataIn.real[initRunLen + trainRunLen:initRunLen + trainRunLen + testRunLen], dataIn.imag[initRunLen + trainRunLen:initRunLen + trainRunLen + testRunLen]]
# Activation configuration
activation_res = "tanh"
activation_out = "tanh"
# Noise level
noiseLevel = 0
# NETWORK TRAINING
net.training(trainInput, trainOutput, initRunLen, trainRunLen, activation_res, activation_out, noiseLevel)
# NETWORK TESTING
dataOut_mat = net.testing(testInput, testRunLen, activation_res, activation_out, noiseLevel)
dataOut_esn = dataOut_mat[:, 0] + 1j * dataOut_mat[:, 1]

"LMS Equalizer"
dataOut_lms = rx.lmsNormalEq(dataOut, dataIn)

"DATA PLOTTING"
plt.figure(1)
plt.plot(dataOut.real, dataOut.imag, "bo")
plt.plot(dataOut_lms.real, dataOut_lms.imag, "yo")
plt.plot(dataOut_esn.real, dataOut_esn.imag, "ro")
plt.plot(dataIn.real, dataIn.imag, "kx")
plt.title('Nonlinear distorted signal (blue) vs LMS (yellow) vs ESN (red)')
plt.show()