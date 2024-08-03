print("Creating Neural Networks and Objects Using Numpy:")
import numpy as np

X = [[1.0,2.0,3.0,3.5],
     [2.0,5.0,-1.0,2.0],
     [-1.5,2.7,3.3,-0.8]
    ]

print("The values of X are:", X)
print("The shape of X is:", np.shape(X))

np.random.seed(0)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10* np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_RELU:
    def forward(self, inputs):
        self.ouptut = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values =(np.exp(inputs - np.max(inputs, axis=1, keepdims = True)))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities 

Layer1 = Layer_Dense(4,3)
Activation1 = Activation_RELU()
Layer2 = Layer_Dense(3,5)
Activation2 = Activation_RELU()
Layer3 = Layer_Dense(5,9)
Activation3 = Activation_RELU()

Layer1.forward(X)
# print("The Output of Layer 1 Is:", Layer1.output)
Layer2.forward(Layer1.output)
# print("The Output of Layer 2 Is:", Layer2.output)
Layer3.forward(Layer2.output)
Activation1.forward(Layer1.output)
Activation2.forward(Layer2.output)
Activation3.forward(Layer3.output)
print("The Output of Layer 3 Is:", Layer3.output)
# print("The Output of Activation 1 Is:", Activation1.ouptut)
# print("The Output of Activation 2 Is:", Activation2.ouptut)
print("The Output of Activation 3 Is:", Activation3.ouptut)


