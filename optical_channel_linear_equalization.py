import numpy as np
import matplotlib.pyplot as plt
import transceiver.transmitter as tx
import transceiver.receiver as rx

"PARAMETER DEFINITION"
M = 16                                                                  # Modulation order
num_symbols = 10000                                                     # No. of symbols
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

"LINEAR EQUALIZER"
dataIn = tx_symbols
dataOut = rx_symbols
dataOut_lms = rx.lmsNormalEq(dataOut, dataIn)

"DATA PLOTTING"
plt.figure(1)
plt.plot(dataOut.real, dataOut.imag, "bo")
plt.plot(dataOut_lms.real, dataOut_lms.imag, "yo")
plt.plot(dataIn.real, dataIn.imag, "rx")
plt.title('Constellation with LMS Filter')
plt.show()