import numpy as np
import sys
import os

# Añadir el directorio padre al path para que Python encuentre el módulo neuron
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuron.neurona import Neuron

class Layer:

    def __init__(self, n_neurons, inputs_size):
        self.neurons = [Neuron(inputs_size) for _ in range(n_neurons)]

    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

    def backward(self, d_output, learning_rate):
        d_inputs = np.zeros(self.neurons[0].inputs)
        for i, neuron in enumerate(self.neurons):
            d_inputs += neuron.backward(d_output[i], learning_rate) 
        return d_inputs

if __name__ == "__main__":
    layer = Layer(3, 4)
    inputs = np.array([1, 4, 6, 8])

    layer_output = layer.forward(inputs)
    print("Layer output:",layer_output)
  