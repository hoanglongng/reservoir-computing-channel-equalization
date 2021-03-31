import numpy as np

def data_generator(bits, symbol_set, SNR_dB):
    """
    Data generated from a wireless communication channel.
    Parameters
    ----------
    bits: Number of bits
    symbol_set: Set of available symbols
    SNR_dB : Signal to noise ratio in decibel
    """
    pre_bits = 7                                                                    # No. of pre-bits involved in ISI
    post_bits = 2                                                                   # No. of post-bits involved in ISI
    w_isi = np.array([0.01, 0.03, 0.04, -0.05, 0.091, -0.1, 0.18, 1, -0.12, 0.08])  # ISI weights
    w_nlr = np.array([1, 0.036, -0.011])                                            # Nonlinear weights

    # Transmitter
    d = np.random.choice(symbol_set, (bits, ))                                      # Transmitted symbols
    d_pad = np.pad(d, (pre_bits, post_bits))                                        # Zero padding

    # ISI and nonlinearity
    q = np.zeros(bits)                                                          
    for n in range(0, len(d_pad) - pre_bits - post_bits):
        q[n] = np.sum(d_pad[n : n + pre_bits + post_bits + 1] * w_isi)
        q[n] = np.sum(np.array([q[n], q[n]**2, q[n]**3]) * w_nlr)

    # Noise
    mean_noise = 0
    P_sig = np.mean(q**2)
    P_sig_dB = 10 * np.log(P_sig)
    P_noise_dB = P_sig_dB - SNR_dB
    P_noise = 10 ** (P_noise_dB / 10)
    std_noise = np.sqrt(P_noise - mean_noise ** 2) 
    n = np.random.normal(mean_noise, std_noise, len(q))
    q = q + n

    return q, d