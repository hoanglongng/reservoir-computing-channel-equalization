import numpy as np
from numpy import matlib as ml
from scipy.signal import upfirdn
from transceiver.transmitter import rrcosine, rcosine

#-----------------------------FILTER-----------------------------
def matched_filter(Psig_Rx, roll_off, span, sps):
    rrcfilter = rrcosine(roll_off, span, sps)
    rxSignal = upfirdn(rrcfilter, Psig_Rx, 1, 1)
    rxSignal_filtered = rxSignal[int(span*sps/2): -int(span*sps/2)]
    return rxSignal_filtered

#-----------------------------SYMBOL DETECTION-----------------------------
def symbol_detector(received_symbols, original_symbols, num_bits, symbol_set):
    received_block = ml.repmat(received_symbols, len(symbol_set), 1)
    reference_block = ml.repmat(symbol_set, num_bits, 1).T
    distance = np.abs(received_block - reference_block)
    index_nearest = np.argmin(distance, axis = 0)
    detected_symbols = symbol_set[index_nearest]
    return detected_symbols

def SER_calculation(detected_symbols, original_symbols, num_bits):
    error_block = detected_symbols - original_symbols
    num_errors = sum(1 for is_error in error_block if is_error != 0)
    SER = num_errors / num_bits
    return SER


#-----------------------------DISTORTION COMPENSATION-----------------------------
def disp_eq(Ein, Disp, Length, w):
    " equalizer of linear dispersion effects"
    c       =  3e8
    lamda0  =  1550*1e-9
    beta2   = -(lamda0**2)*Disp/(2*np.pi*c)
    TF_DispFiber = np.exp(-1j*beta2*w**2/2*Length)
    PF      =  fftshift(fft(Ein))
    return ifft(ifftshift(PF*TF_DispFiber))

def LMSNormalEq(sig, ref, width = 8):
    " perform a linear lms filter by solving Normal Equations "
    X = window_stack(sig, width=8)
    Rm = inv(np.dot(X.conj().T,X)).dot(X.conj().T).dot(ref.reshape(-1,1))
    return X.dot(Rm)

def lmsNormalEq(x, y):
    x_real = x.real
    y_real = y.real
    x_imag = x.imag
    y_imag = y.imag
    X = np.c_[x_real, x_imag]
    Y = np.c_[y_real, y_imag]
    theta0 = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    Y = X.dot(theta0)
    return Y[:, 0] + 1j * Y[:, 1]

#-----------------------------RECEIVER-----------------------------
def power_denorm(Ein, Psig, Pref):
    "Power denormalization from a given reference power level"
    Psig_Rx    = Ein * np.sqrt(Psig)/np.sqrt(Pref)
    return Psig_Rx

def QAM(Eout, roll_off, span, sps, Psig, Pref):
    "Receiver with QAM format"
    Psig_Rx = power_denorm(Eout, Psig, Pref)
    rxSignal = matched_filter(Psig_Rx, roll_off, span, sps)
    x = np.arange(0, len(rxSignal), step = sps)
    dataOut = rxSignal[x]
    return dataOut