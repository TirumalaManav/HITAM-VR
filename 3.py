import numpy as np

print("Building Basic Neural Layer Using Numpy: ")

inputs = [1.0,2.0,3.0,2.5]

weights = [[0.2,0.8,-0.5,1.0],
              [0.5,-0.91,0.26,-0.5],
              [-0.26,-0.27,0.17,0.87]]

bias = [2.0,3.0,0.5]
output = np.dot(weights,inputs) + bias

print("OUTPUT of Basic Neural Layer:", output)

