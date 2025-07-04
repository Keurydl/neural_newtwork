import numpy as np

class Neuron:
    def __init__(self, n_input):
        self.weights = np.random.randn(n_input)
        self.bias = np.random.randn()
        self.output = 0
        self.inputs = None
        self.dweights = np.zeros_like(self.weights)
        self.dbias = 0

    def activate(self, x):
        return 1 / (1 + np.exp(-x))

    def derivate_activate(self, x):
        return x * (1 - x)


    def forward(self, inputs):
        self.inputs = inputs
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        self.output = self.activate(weighted_sum)
        return self.output

    def backward(self, d_output, learning_rate):
        d_activation = d_output * self.derivate_activate(self.output)
        self.dweights = np.dot(self.inputs.T, d_activation)
        self.dbias = d_activation
        d_input = np.dot(d_activation, self.weights)
        self.weights -= self.dweights * learning_rate
        self.bias -= learning_rate * self.dbias
        return d_input
        
if __name__ == "__main__":
    # Crear una neurona con 3 entradas
    neuron = Neuron(3)
    
    # Definir las entradas de ejemplo
    inputs = np.array([0.5, 0.3, 0.9])
    
    print("Inputs:", inputs)
    print("Pesos iniciales:", neuron.weights)
    print("Bias inicial:", neuron.bias)
    
    # Calcular la salida
    output = neuron.forward(inputs)
    
    print("\nSalida de la neurona:", output)

