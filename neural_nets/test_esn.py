import numpy as np
import matplotlib.pyplot as plt
from esn import ESN

# RESERVOIR GENERATION
inSize = 1
outSize = 1
resSize = 1000
connectivity = 0.01
spectralRadius = 0.8
inputScaling = 1
withFeedback = 1
net = ESN(inSize, outSize, resSize, connectivity, spectralRadius, inputScaling, withFeedback)

def generate(sampleLen, initWashoutLen):
    tau = 17
    incPerUnit = 10
    genHistoryLen = tau * incPerUnit
    seed = 1.2 * np.ones((genHistoryLen, 1)) + 0.2 * (np.random.rand(genHistoryLen, 1) - 0.5)
    oldVal = 1.2
    genHistory = seed
    speedup = 1
    sample = np.zeros((sampleLen, 1))
    step = 0
    for n in range(1, sampleLen + initWashoutLen):
        for i in range(1, incPerUnit * speedup):
            step = step + 1
            tauVal = genHistory[step % genHistoryLen]
            newVal = oldVal + (0.2 * tauVal / (1.0 + tauVal**10) - 0.1 * oldVal) / incPerUnit
            genHistory[step % genHistoryLen] = oldVal
            oldVal = newVal
    
        if n > initWashoutLen:
            sample[n - initWashoutLen] = newVal

    trainseq = sample[0:sampleLen-1] - 1.0
    trainseq = np.tanh(trainseq)
    return trainseq

# TRAINING INITIALIZATION
# Sample loading
trainData = np.loadtxt('neural_nets/MackeyGlass_t17.txt')
# Training data division
initRunLen = 1000
trainRunLen = 2000
testRunLen = 2000
# Input - Output definition
inTraining = 0.02 * np.ones(initRunLen + trainRunLen)
outTraining = trainData[0:initRunLen + trainRunLen]
inTesting = 0.02 * np.ones(testRunLen)
outTesting = trainData[initRunLen + trainRunLen: initRunLen + trainRunLen + testRunLen]
# Activation configuration
activation_res = "tanh"           # Activation function in the reservoir
activation_out = "tanh"           # Activation function at the output
# Noise level
noiseLevel = 1e-10

# NETWORK TRAINING
net.training(inTraining, outTraining, initRunLen, trainRunLen, activation_res, activation_out, noiseLevel)

# NETWORK TESTING
netOutTest = net.testing(inTesting, testRunLen, activation_res, activation_out, noiseLevel)

# DATA PLOTTING
plt.figure(1)
plt.plot(outTesting)
plt.plot(netOutTest)
plt.gca().legend(('Actual', 'Predicted'))
plt.show()