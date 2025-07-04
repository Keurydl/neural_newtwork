# Multilayer Neural Network (MLP) from Scratch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational implementation of a multilayer neural network (MLP) from scratch in Python, with an intuitive graphical interface for experimenting with different network configurations and datasets.

## 🚀 Features

- Implementation of multilayer neural networks (MLP) from scratch
- Intuitive graphical interface with Tkinter
- Real-time training visualization
- Easy to use and modify
- Ideal for learning and experimentation

## 📦 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## 🛠️ Installation

1. Clone this repository:
```bash
git clone https://github.com/your-user/mlp-neural-network.git
cd mlp-neural-network
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install them manually:
```bash
pip install numpy matplotlib
```

## 🎮 Basic Usage

1. Start the graphical application:
```bash
python app.py
```

2. Configure your neural network:
- Add layers with the desired number of neurons
- Set the input size for the first layer

3. Add training data:
- Enter the input data separated by commas
- Enter the expected labels

4. Train the network:
- Set the number of epochs and the learning rate
- Click "Train Network"

5. Test the network with new data

## 🧠 Examples

### OR Gate

1. Network configuration:
- Input layer: 2 neurons
- Hidden layer: 4 neurons
- Output layer: 1 neuron

2. Input data Training:
- Inputs: [0,0], [0,1], [1,0], [1,1]
- Outputs: [0], [1], [1], [1]

3. Parameters:
- Epochs: 1000
- Learning Rate: 0.1

### AND Logic Gate

Uses the same configuration as OR, but with outputs: [0], [0], [0], [1]

## 🏗️ Project Structure

```
neural-network-mlp/
├── app.py # Main graphical interface
├── README.md # This file
├── requirements.txt # Dependencies
├── neuron/ # Neuron module
│ ├── __init__.py
│ └── neuron.py # Neuron implementation
├── layer/ # Layer module
│ ├── __init__.py
│ └── layer.py # Layer implementation
└── neural_network/ # Neural network module
├── __init__.py
└── neural_network.py # MLP network implementation
```

## 📚 Technical Documentation

### Neuron
The `Neuron` class implements a single neuron with:
- Random weights and bias
- Sigmoid activation function
- Error backpropagation

### Layer
The `Layer` class manages a layer of neurons:
- Multiple neuron initialization
- Forward propagation
- Backpropagation of the error

### NeuralNetwork
The `NeuralNetwork` class implements a complete MLP network:
- Add multiple layers
- Train with backpropagation
- Predict with new data

## 🤝 Contributions

Contributions are welcome! Please read our [contribution guide](CONTRIBUTING.md) for more details.

## View:

![image](https://github.com/user-attachments/assets/05de0c54-b012-47b6-8e7b-cea68b672c0b)

<div align="center">
Made with ❤️ by [Keury]
</div>
