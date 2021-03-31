import numpy as np
from math import pi
from scipy.signal import upfirdn

#-----------------------------SIGNAL CODING-----------------------------
def bi2de(x):
    "Convert a binary matrix to a row of non-negative decimal integers"
    n = x.shape[0]
    outMatrix = np.zeros([n], dtype = np.int)
    for k in range(n):
        outMatrix[k] = sum(1<<i for (i,x) in enumerate(x[k]) if x !=0)
    return outMatrix
 
def symbol_generation(num_symbols, M):
    "Generate a row of random symbol words corresponding to random input bit stream"
    bits_per_symbol = int(np.log2(M)) 
    num_bits = num_symbols * bits_per_symbol
    dataInMatrix = np.reshape(np.random.randint(2, size = num_bits), (num_symbols, bits_per_symbol)) 
    symbol_words = bi2de(dataInMatrix)                                                           
    return symbol_words   

#-----------------------------PULSE SHAPING-----------------------------
def rrcosine(beta, span, sps):
    "Impulse response of a square root raised cosine filter"
    N = span * sps 
    sample_num =  np.arange(N+1)
    h_rrc = np.zeros(N+1, dtype = float)
    Ts = sps
    for x in sample_num:
        t = (x - N/2)
        if t == 0.0:
            h_rrc[x] = (1/np.sqrt(Ts))*(1.0 - beta + (4*beta/np.pi))
        elif (beta != 0) and (t == Ts/(4*beta) or t == -Ts/(4*beta)):
            h_rrc[x] = (1/np.sqrt(Ts))*(beta/(np.sqrt(2)))*(((1+2/np.pi)*(np.sin(np.pi/(4*beta)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*beta)))))
        else:
            h_rrc[x] = (1/np.sqrt(Ts))*(np.sin(np.pi*t*(1-beta)/Ts) + 4*beta*(t/Ts)*np.cos(np.pi*t*(1+beta)/Ts))/(np.pi*t*(1-(4*beta*t/Ts)*(4*beta*t/Ts))/Ts)
    return h_rrc

def rcosine(beta, span, sps, norm = False):
    "Impulse response of a raised cosine filter"
    N = span * sps 
    sample_num =  np.arange(N+1)
    h_rc = np.zeros(N+1, dtype = float)
    Ts = sps
    for x in sample_num:
        t = (x - N/2)
        if (beta != 0) and (t == Ts/(2*beta) or t == -Ts/(2*beta)):
            h_rc[x] = (pi/(4*Ts))*np.sinc(1/(2*beta))
        else:
            h_rc[x] = (1/Ts)*np.sinc(t/Ts)*np.cos((pi*beta*t)/Ts)/(1-((2*beta*t)/Ts)**2)
    if norm:
        h_rc = h_rc / np.amax(h_rc)
    return h_rc

def gaussianfilter(bt, span, sps):
    "Impulse response of a gaussian filter"
    alpha = sps*bt
    N          = span*sps        
    Ts         = sps              
    h_gaussian = np.zeros(N+1, dtype = float)
    time_idx   = np.arange(N+1)-N/2
    h_gaussian = (np.sqrt(pi)/alpha)*np.exp(-(pi*time_idx/alpha)**2)
    return time_idx, h_gaussian

def pulse_shaping(dataIn, roll_off, span, sps):
    "General module performing pulse shaping task"
    rrcfilter = rrcosine(roll_off, span, sps)
    txSignal = upfirdn(rrcfilter, dataIn, sps)
    txSignal_filtered = txSignal[int(span*sps/2): -int(span*sps/2)]
    return txSignal_filtered

#-----------------------------MODULATION-----------------------------
def QAM_modulation(symbolWord, M):
    "Modulation in QAM format given a vector of input symbol"
    bits_per_symbol = int(np.sqrt(M))
    real = np.transpose(np.tile(np.linspace(-bits_per_symbol + 1, bits_per_symbol - 1, bits_per_symbol, dtype = int), (4, 1)))
    imag = np.tile((np.linspace(-bits_per_symbol + 1, bits_per_symbol - 1, bits_per_symbol, dtype = int) * 1j), (4, 1))
    const = (real + imag).flatten()
    data = const[symbolWord]
    return data, const

#-----------------------------TRANSMITTER-----------------------------
def power_norm(input_signal, Pref):
    "Power normalization to a given reference power level"
    Psig = (np.abs(input_signal)**2).mean()
    output_signal = np.sqrt(Pref) * input_signal / np.sqrt(Psig)
    return output_signal, Psig

def QAM(num_symbols, M, roll_off, span, sps, Pref):
    "Transmitter with QAM format"
    symbol_words = symbol_generation(num_symbols, M)
    tx_symbols, _ = QAM_modulation(symbol_words, M)                  
    txSignal = pulse_shaping(tx_symbols, roll_off, span, sps)
    Ein, Psig = power_norm(txSignal, Pref)
    return Ein, tx_symbols, Psig

