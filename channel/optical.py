import numpy as np

def SMF(Ein, length, gamma, dispersion, loss_dB, w, max_phase_change, max_step_width):
    """
    Data generated from a single-mode fibre (SMF) using Split-Step Fourier (SSF) method
    -----------
    Parameters:
    Ein: Input field
    length: Fibre length
    gamma: Nonlinearity coefficient
    dispersion: Dispersion coefficient
    loss_dB: Attenuation in dB
    w: Circular frequency axis
    max_phase_change: Maximum phase change
    max_step_width: Maximum step width
    """
    z0 = 0
    c  = 3e8
    lambd = 1550e-9
    alpha = loss_dB/(10*log10(e))
    beta2 = -(lambd**2)*dispersion/(2*pi*c)
    
    dph_max  = max_phase_change
    dz_phase = dph_max/(gamma*(np.abs(Ein)**2).max())
    dz       = min(dz_phase, max_step_width)
    
    PF = np.zeros(w.shape)
    TF_dispersionFiber = np.zeros(w.shape)
    E_out_disp   = np.zeros(Ein.shape)
    E_out_nl     = np.zeros(Ein.shape)

    while (z0+dz) < length:
        # First step (attenuation and dispersion added in frequency domain)
        PF          = fftshift(fft(Ein))
        attenuation = np.exp(-(alpha/2) * (dz/2))
        dispersion  = np.exp(1j*beta2*w**2/2*(dz/2))
        E_out_disp  = ifft(ifftshift(PF * dispersion * attenuation))
        
        # Second step (nonlinear added in time domain)
        nonlinear   = np.exp(1j*gamma*dz*np.abs(E_out_disp)**2)
        E_out_nl    = E_out_disp * nonlinear
        
        # Final step (attenuation and dispersion added in frequency domain)
        PF          = fftshift(fft(E_out_nl))
        E_out_disp  = ifft(ifftshift(PF * dispersion * attenuation))
        
        # Calculation parameters for the next step
        z0      += dz
        Ein      = E_out_disp
        dz_phase = dph_max/(gamma*(np.abs(Ein)**2).max())
        dz       = min(dz_phase, MaxStepWidth)
        
    # Final Step
    dz = length - z0
    
    # First step (dispersive) of Fourier
    PF            = fftshift(fft(Ein))
    attenuation = np.exp(-(alpha/2) * (dz/2))
    dispersion  = np.exp(1j*beta2*w**2/2*(dz/2))
    E_out_disp = ifft(ifftshift(PF*dispersion*attenuation))

    # Second step (nonlinear section)
    E_out_nl    = E_out_disp*np.exp(1j*gamma*dz*np.abs(E_out_disp)**2)

    # Final step (dispersive) of Fourier
    PF            = fftshift(fft(E_out_nl))
    E_out_disp = ifft(ifftshift(PF*dispersion*attenuation))

    return E_out_disp

def awgn(data, snr_db):
    """
    Adding noise to the input signal given SNR level
    -----------
    Parameters:
    data: Input signal
    snr_db: Signal to noise ratio in dB
    """
    snr = 10 ** (snr_db/10)
    P = np.sum(abs(data)**2) / len(data)
    N0 = P / snr
    noise = np.sqrt(N0 / 2) * (np.random.standard_normal(data.shape) + 1j * np.random.standard_normal(data.shape))
    return data + noise

def AmpCG(Ein, Gain, NF, SampleRate, NSamples):
    """
    Amplifier with given gain and noise figure level
    -----------
    Parameters:
    Ein: Input signal
    Gain: Gain level
    NF: Noise figure level
    SampleRate: Signal sample rate
    NSamples: Number of signal samples
    """
    h  = 6.63e-34
    fc = 193.1e12
    No = h*fc*(Gain-1)*NF/2
    noise = np.sqrt(No*SampleRate/2)*(np.random.randn(NSamples) + 1j*np.random.randn(NSamples))
         
    return np.sqrt(Gain)*Ein + noise