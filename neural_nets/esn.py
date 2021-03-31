import numpy as np
from neural_nets.activations import RFT
from scipy.sparse import random, linalg

class ESN:

    def __init__(self, inSize, outSize, resSize, connectivity, spectralRadius, inputScaling, withFeedback):
        self.inSize = inSize
        self.outSize = outSize
        self.resSize = resSize
        self.connectivity = connectivity
        self.spectralRadius = spectralRadius
        self.inputScaling = inputScaling
        self.withFeedback = withFeedback
        self.totalDim = self.inSize + self.outSize + self.resSize
        self.totalState = np.zeros((self.totalDim, 1))
        self.internalState = self.totalState[0:self.resSize]
        self.initInWM()
        self.initIntWM()
        self.initOutWM()
        if self.withFeedback:
            self.initOfbWM()

    def initInWM(self):
        # Input weight matrix initialized with uniformly distributed over [-1;1]
        self.inWM = self.inputScaling * (2.0 * np.random.rand(self.resSize, self.inSize) - 1.0)

    def initIntWM(self):
        # Internal weight matrix sparsely initialized with uniformly distributed over [-0.5;0.5]
        self.intWM = random(self.resSize, self.resSize, self.connectivity).toarray() - 0.5
        self.intWM[self.intWM == -0.5] = 0
        self.maxval = np.max(np.abs(np.linalg.eigvals(self.intWM)))
        self.intWM = self.intWM * self.spectralRadius / self.maxval

    def initOutWM(self):
        # Output weight matrix initialized by blank for training purpose
        self.outWM = np.zeros((self.outSize, self.resSize + self.inSize))

    def initOfbWM(self):
        # Output feedback weight matrix initialized with uniformly distributed over [-1;1]
        self.ofbWM = 2.0 * np.random.rand(self.resSize, self.outSize) - 1.0 

    def training(self, trainDataInput, trainDataOutput, initRunLen, trainRunLen, activation_res, activation_out, noiseLevel):
        self.internalState = self.totalState[0:self.resSize]
        # Collect state matrices over entire training process
        stateCollectMat = np.zeros((trainRunLen, self.resSize + self.inSize))
        teachCollectMat = np.zeros((trainRunLen, self.outSize))

        # TRAINING ITERATION
        print('Start learning...')
        for i in range(1, initRunLen + trainRunLen + 1):

            # Input node
            netIn = np.transpose(trainDataInput[[i-1]])

            # Output teacher
            if activation_out == "linear":
                teach = np.transpose(trainDataOutput[[i-1]])
            elif activation_out == "tanh":
                teacher = np.transpose(trainDataOutput[[i-1]])
                teach = np.tanh(teacher)
            elif activation_out == "RFT":
                teacher = np.transpose(trainDataOutput[[i-1]])
                teach = RFT(teacher)

            # Writing input to totalState
            if self.inSize > 0:
                self.totalState[self.resSize:self.resSize + self.inSize, :] = netIn

            # Creating a matrix of all network weights
            totalWeights = np.zeros((self.resSize, self.totalDim))
            totalWeights[0:self.resSize, 0:self.resSize] = self.intWM
            totalWeights[0:self.resSize, self.resSize:(self.resSize + self.inSize)] = self.inWM
            if self.withFeedback:
                totalWeights[0:self.resSize, (self.resSize + self.inSize):self.totalDim] = self.ofbWM  

            # Update internal states
            if activation_res == "linear":
                self.internalState = np.matmul(totalWeights, self.totalState) + noiseLevel * 2.0 * (np.random.rand(self.resSize, 1) - 0.5)
            elif activation_res == "tanh":
                self.internalState = np.tanh(np.matmul(totalWeights, self.totalState) + noiseLevel * 2.0 * (np.random.rand(self.resSize, 1) - 0.5))
            elif activation_res == "RFT":
                self.internalState = RFT(np.matmul(totalWeights, self.totalState) + noiseLevel * 2.0 * (np.random.rand(self.resSize, 1) - 0.5))
            self.totalState[0:self.resSize, :] = self.internalState

            # Update output 
            if activation_out == "linear":
                netOut = np.matmul(self.outWM, self.totalState[0:(self.resSize + self.inSize), :])
            elif activation_out == "tanh":    
                netOut = np.tanh(np.matmul(self.outWM, self.totalState[0:(self.resSize + self.inSize), :]))
            elif activation_out == "RFT": 
                netOut = RFT(np.matmul(self.outWM, self.totalState[0:(self.resSize + self.inSize), :]))
            self.totalState[(self.resSize + self.inSize):self.totalDim, :] = netOut

            # Forcing teacher output (During washout and training periods)
            if i <= initRunLen + trainRunLen:
                self.totalState[(self.resSize + self.inSize):self.totalDim, :] = teach

            # Collecting states for later use in computing model (During training periods)
            if (i > initRunLen) and (i <= initRunLen + trainRunLen):
                collectIndex = i - initRunLen - 1
                stateCollectMat[collectIndex, 0:self.resSize] = np.transpose(self.internalState)
                stateCollectMat[collectIndex, self.resSize:self.resSize+self.inSize] = np.transpose(netIn)
                if activation_out == "linear":
                    teachCollectMat[collectIndex, :] = np.transpose(teach)
                elif activation_out == "tanh" or activation_out == "RFT":    
                    teachCollectMat[collectIndex, :] = np.transpose(teacher)

            # Computing new model (At the end of the training period)
            if i == (initRunLen + trainRunLen):
                self.outWM = np.transpose(np.matmul(np.linalg.pinv(stateCollectMat), teachCollectMat))
                msetrain = np.sum(np.square(np.tanh(teachCollectMat) - np.tanh(np.matmul(stateCollectMat, np.transpose(self.outWM)))))
                print('MSE_train =', msetrain)
        print('Learning completed!')

    def testing(self, trainDataIn, testRunLen, activation_res, activation_out, noiseLevel):
        # State matrices
        self.internalState = self.totalState[0:self.resSize]
        # Collect netout matrix
        netOutMat = np.zeros((testRunLen, self.outSize))

        # TRAINING ITERATION
        print('Start testing...')
        for i in range(1, testRunLen+1):

            # Input node
            netIn = np.transpose(trainDataIn[[i-1]])

            # Writing input to totalState
            if self.inSize > 0:
                self.totalState[self.resSize:self.resSize + self.inSize, :] = netIn

            # Creating a matrix of all network weights
            totalWeights = np.zeros((self.resSize, self.totalDim))
            totalWeights[0:self.resSize, 0:self.resSize] = self.intWM
            totalWeights[0:self.resSize, self.resSize:(self.resSize + self.inSize)] = self.inWM
            if self.withFeedback:
                totalWeights[0:self.resSize, (self.resSize + self.inSize):self.totalDim] = self.ofbWM  

            # Update internal states
            if activation_res == "linear":
                self.internalState = np.matmul(totalWeights, self.totalState) + noiseLevel * 2.0 * (np.random.rand(self.resSize, 1) - 0.5)
            elif activation_res == "tanh":
                self.internalState = np.tanh(np.matmul(totalWeights, self.totalState) + noiseLevel * 2.0 * (np.random.rand(self.resSize, 1) - 0.5))
            elif activation_res == "RFT":
                self.internalState = RFT(np.matmul(totalWeights, self.totalState) + noiseLevel * 2.0 * (np.random.rand(self.resSize, 1) - 0.5))
            self.totalState[0:self.resSize, :] = self.internalState

            # Update output 
            if activation_out == "linear":
                out = np.matmul(self.outWM, self.totalState[0:(self.resSize + self.inSize), :])
                netOut = out
            elif activation_out == "tanh":
                out = np.matmul(self.outWM, self.totalState[0:(self.resSize + self.inSize), :])    
                netOut = np.tanh(out)
            elif activation_out == "RFT":
                out = np.matmul(self.outWM, self.totalState[0:(self.resSize + self.inSize), :]) 
                netOut = RFT(out)
            self.totalState[(self.resSize + self.inSize):self.totalDim, :] = netOut

            # Collect network output
            netOutMat[[i-1]] = np.transpose(out)

        print('Testing completed!')
        return netOutMat