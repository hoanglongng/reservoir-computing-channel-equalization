import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn
import transceiver.transmitter as tx
import transceiver.receiver as rx
import neural_nets.esn_simplified as rc

"PARAMETER DEFINITION"
M = 16                                                                  # Modulation order
symbols = 2**15                                                         # No. of symbols
sps = 16                                                                # No. of samples/symbol
baudRate = 32e9                                                         # Symbol rate
bitsPerSymbol = int(np.log2(M))                                         # No. of bits/symbol
bits = symbols * bitsPerSymbol                                          # No. of bits
sampleRate = sps * baudRate                                             # Sample rate
simulWind = symbols / baudRate                                          # Simulation window
samples = sps * symbols                                                 # No. of samples
ws = 2 * np.pi * sampleRate                                             # Sample rate in rad/s
time = np.linspace(0, simulWind - 1/sampleRate, num = samples)          # Time axis of sampled points
w = ws * np.arange(-samples/2, samples/2)/samples                       # Frequency axis
Pref_dBm = 10
Pref = 10 ** (Pref_dBm/10) * 1e-3

"FIBER PARAMETERS"
loss = 0.2e-3
dispersion = 0
lengthKm = 100
length = lengthKm * 1e3
gamma = 1e-4
maxPhaseChange = 0.005
maxStepWidth = 1000
Gain_dB  = loss*length
Gain     = 10**(Gain_dB/10)
NF_dB    = 4.5
NF       = 10**(NF_dB/10)  

"SIGNAL TRANSMITTER"
dataInMatrix = np.reshape(np.random.randint(2, size = bits), (symbols, bitsPerSymbol))      # Generate bit stream 
symbolWords = tx.bi2de(dataInMatrix)                                                           # Matrix of symbol values
dataIn, alphabet = tx.qam_modulate(symbolWords, M)                                             # QAM modulation
roll_off = 0.1
span = 1024
rrcfilter = tx.rrcosine(roll_off, span, sps)
txSignal = upfirdn(rrcfilter, dataIn, sps)
txSignal = txSignal[int(span*sps/2):int(span*sps/2)+samples]
Psig = (np.abs(txSignal)**2).mean()
Ein = np.sqrt(Pref) * txSignal / np.sqrt(Psig)

"SIGNAL TRANSMISSION"
# Nonlinear channel (zero dispersion)
Eout = Ein * np.exp(1j * gamma * length * np.abs(Ein)**2)
# Eout = Ein
# Eout = SMF_SSF(Ein, length, gamma, dispersion, loss, w, maxPhaseChange, maxStepWidth)
# Eout = AmpCG(Eout, Gain, NF, sampleRate, samples)

"SIGNAL RECEIVER USING ESN APPROACH"
# # Dispersion equalizer
# E_disp_eq  = disp_eq(Eout, dispersion, length, w)
Psig_Rx    = Eout*np.sqrt(Psig)/np.sqrt(Pref)
# Matched filter
rxSignal = upfirdn(rrcfilter, Psig_Rx, 1, 1)
rxSignal = rxSignal[int(span*sps/2):int(span*sps/2)+samples]
# Symbol detector
x = np.arange(0, len(rxSignal), step = 16)
dataOut = rxSignal[x]

"RC EQUALIZATION"
train_len = 1000
dataIn_train = dataOut[: train_len]
dataOut_train = dataIn[: train_len]
dataIn_test = dataOut[train_len: ]
dataOut_test = dataIn[train_len: ]
print(len(dataOut_test))
num_nodes = 100
beta = 1
alpha = 0.2
act_func = "tanh"
# Training
theta0_real, weights_samples_real = rc.training(dataIn_train.real, dataOut_train.real, num_nodes, alpha, beta, act_func)
theta0_imag, weights_samples_imag = rc.training(dataIn_train.imag, dataOut_train.imag, num_nodes, alpha, beta, act_func)
# Testing
Y_hat_real = rc.testing(dataIn_test.real, theta0_real, weights_samples_real, num_nodes, alpha, beta, act_func)
Y_hat_imag = rc.testing(dataIn_test.imag, theta0_imag, weights_samples_imag, num_nodes, alpha, beta, act_func)

"LMS"
dataOut_lms = rx.lmsNormalEq(dataIn_test, dataOut_test)

plt.figure(1)
plt.plot(dataIn_test.real, dataIn_test.imag, "bo")
plt.plot(Y_hat_real, Y_hat_imag, 'ro')
plt.plot(dataOut_test.real, dataOut_test.imag, "kx")
plt.title('Nonlinear distorted signal (blue) vs RC equalization (red)')
plt.figure(2)
plt.plot(dataIn_test.real, dataIn_test.imag, "bo")
plt.plot(dataOut_lms.real, dataOut_lms.imag, 'ro')
plt.plot(dataOut_test.real, dataOut_test.imag, "kx")
plt.title('LMS')
plt.show()