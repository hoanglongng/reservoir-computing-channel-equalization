This folder contains files of different neural network architectures. These can be used for equalization in optical communication.

***MAIN FILES***
activations.py
NONLINEAR ACTIVATION FUNCTIONS
Consisting of functions used for neural networks. They can be conventional activation functions (e.g. tanh, sigmoid, ReLU, Leaky ReLU) or nonlinear transfer function of optical elements (e.g. SOA)

esn.py
FULL ECHO STATE NETWORK (ESN)
Having full architecture of an ESN where reservoir has multiple nodes sparsely connected and feedback from output layer (if withFeedback == 1). Output is computed over weighted input and internal nodes.

esn_simplified.py
SIMPLIFIED ECHO STATE NETWORK (ESN)
Having simplified architecture of an ESN with internal nodes connected in ring topology. It is suitable for optical implementation of ESN so it will be used to model reservoir computing experimental setup.

linear_regression.py
LINEAR REGRESSION CLASSIFIER
Using normal equation to find parameters of linear regression. It will be used as linear equalizer of optical communication.

***TEST FILE***
test_activations.py
TESTING OF ACTIVATION FUNCTIONS
Plotting nonlinear activation functions, both conventional and optical ones.

test_esn.py
TESTING OF ESN
Testing the performance of ESN in Mackey-Glass time-series predictions.

MackeyGlass_t17.txt
Containing Mackey-Glass data for testing ESN architecture.