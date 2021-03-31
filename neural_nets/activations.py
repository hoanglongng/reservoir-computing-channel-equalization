import numpy as np
from math import pi
from scipy.optimize import fsolve

"CONVENTIONAL ACTIVATION FUNCTIONS"
# Tanh
# np.tanh (numpy built-in function)

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU
def leaky_relu(x):
    return np.where(x >= 0, x, 0.01*x)
    
"OPTICAL ACTIVATION FUNCTIONS"
# Semiconductor optical amplifier (SOA)
def SOA(Pin, I = 0.520, L = 500e-6, tauc = 300e-12, Psat = 0.0211, Gamma = 0.3, a = 2.7e-20, N0 = 1e24, I0 = 0.1067, alpha = 5):
    """
    Instantaneous transfer function of SOA
    -----------
    Parameters:
    I: Input current (A)
    L: Amplifier length (m)
    tauc: Carrier lifetime (s)
    Psat: Saturation power (W)
    Gamma: Confinement factor
    a: Differential gain (m**2)
    N0: Carrier transparent density (1/m**3)
    I0: Transparent current (A)
    alpha: Linewidth enhancement factor (for phase shift)
    """
    Esat = Psat*tauc                                    # Saturation energy (J)
    g0 = Gamma*a*N0*(I/I0 - 1)                          # Linear gain (1/m) 
    # Gain equation in the steady state
    def equation(h):
        return ((g0*L - h)/tauc - Pin*(np.exp(h) - 1)/Esat)
    # Solve above equation and calculate Pout
    h_root = fsolve(equation, 1)
    gain = np.exp(h_root)
    Pout = Pin*gain
    return Pout

# Regenerative Fourier transform
def RFT(x, alpha = 1/pi, beta = pi):
    return x + alpha*np.sin(beta*x)