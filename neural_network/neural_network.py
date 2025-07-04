import numpy as np
import sys
import os

# Asegurarse de que el directorio padre esté en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer.layer import Layer

class NauralNetwork:
    """
    Implementación de una red neuronal multicapa (MLP) desde cero.
    """
    def __init__(self):
        """Inicializa una nueva red neuronal sin capas."""
        self.layers = []
        self.loss_list = []
        self.input_size = None

    def add_layer(self, num_neurons, inputs_size):
        """
        Añade una nueva capa a la red.
        
        Args:
            num_neurons (int): Número de neuronas en la capa.
            inputs_size (int): Tamaño de la entrada para esta capa.
        """
        if not self.layers and self.input_size is None:
            self.input_size = inputs_size
        
        if not self.layers:
            # Primera capa oculta
            self.layers.append(Layer(num_neurons, inputs_size))
        else:
            # Capas posteriores toman como entrada el tamaño de la capa anterior
            previous_output_size = len(self.layers[-1].neurons)
            self.layers.append(Layer(num_neurons, previous_output_size))
    
    def forward(self, inputs):
        """
        Realiza una pasada hacia adelante a través de la red.
        
        Args:
            inputs (numpy.ndarray): Vector de entrada.
            
        Returns:
            numpy.ndarray: Salida de la red.
        """
        if not self.layers:
            raise ValueError("No hay capas en la red")
            
        current_input = np.array(inputs, dtype=float)
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

    def backward(self, loss_gradient, learning_rate):
        """
        Realiza la retropropagación del error.
        
        Args:
            loss_gradient (numpy.ndarray): Gradiente de la función de pérdida.
            learning_rate (float): Tasa de aprendizaje.
            
        Returns:
            numpy.ndarray: Gradiente de la capa de entrada.
        """
        d_input = loss_gradient
        for layer in reversed(self.layers):
            d_input = layer.backward(d_input, learning_rate)
        return d_input

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        """
        Entrena la red neuronal con los datos proporcionados.
        
        Args:
            X (numpy.ndarray): Datos de entrenamiento.
            y (numpy.ndarray): Etiquetas de entrenamiento.
            epochs (int): Número de épocas de entrenamiento.
            learning_rate (float): Tasa de aprendizaje.
        """
        self.loss_list = []
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for i in range(len(X)):
                # Forward pass
                output = self.forward(X[i])
                
                # Calcular pérdida (error cuadrático medio)
                loss = np.mean((y[i] - output) ** 2)
                epoch_loss += loss
                
                # Backward pass
                error = 2 * (output - y[i]) / len(y[i]) if len(y.shape) > 1 else 2 * (output - y[i])
                self.backward(error, learning_rate)
            
            # Calcular pérdida promedio de la época
            epoch_loss /= len(X)
            self.loss_list.append(epoch_loss)
            
            # Imprimir progreso cada 100 épocas
            if (epoch + 1) % 100 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"Época {epoch + 1}/{epochs}, Pérdida: {epoch_loss:.6f}")

    def predict(self, X):
        """
        Realiza predicciones con la red entrenada.
        
        Args:
            X (numpy.ndarray): Datos de entrada para predecir.
            
        Returns:
            numpy.ndarray: Predicciones de la red.
        """
        if not self.layers:
            raise ValueError("La red no tiene capas")
            
        X = np.array(X, dtype=float)
        if X.ndim == 1:
            return self.forward(X)
        return np.array([self.forward(x) for x in X])
    
    def get_weights(self):
        """Obtiene los pesos de todas las capas."""
        return [layer.weights for layer in self.layers]
    
    def get_biases(self):
        """Obtiene los sesgos de todas las capas."""
        return [neuron.bias for layer in self.layers for neuron in layer.neurons]
