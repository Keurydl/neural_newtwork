# Red Neuronal Multicapa (MLP) desde Cero

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Una implementación didáctica de una red neuronal multicapa (MLP) desde cero en Python, con una interfaz gráfica intuitiva para experimentar con diferentes configuraciones de red y conjuntos de datos.

## 🚀 Características

- Implementación desde cero de redes neuronales multicapa (MLP)
- Interfaz gráfica intuitiva con Tkinter
- Visualización en tiempo real del entrenamiento
- Fácil de usar y modificar
- Ideal para aprendizaje y experimentación

## 📦 Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## 🛠️ Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/red-neuronal-mlp.git
   cd red-neuronal-mlp
   ```

2. Crea un entorno virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: .\venv\Scripts\activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

   O instálalas manualmente:
   ```bash
   pip install numpy matplotlib
   ```

## 🎮 Uso Básico

1. Inicia la aplicación gráfica:
   ```bash
   python app.py
   ```

2. Configura tu red neuronal:
   - Añade capas con el número de neuronas deseado
   - Establece el tamaño de entrada para la primera capa

3. Añade datos de entrenamiento:
   - Ingresa los datos de entrada separados por comas
   - Ingresa las etiquetas esperadas

4. Entrena la red:
   - Establece el número de épocas y la tasa de aprendizaje
   - Haz clic en "Entrenar Red"

5. Prueba la red con nuevos datos

## 🧠 Ejemplos

### Compuerta Lógica OR

1. Configuración de la red:
   - Capa de entrada: 2 neuronas
   - Capa oculta: 4 neuronas
   - Capa de salida: 1 neurona

2. Datos de entrenamiento:
   - Entradas: [0,0], [0,1], [1,0], [1,1]
   - Salidas: [0], [1], [1], [1]

3. Parámetros:
   - Épocas: 1000
   - Tasa de aprendizaje: 0.1

### Compuerta Lógica AND

Usa la misma configuración que OR, pero con salidas: [0], [0], [0], [1]

## 🏗️ Estructura del Proyecto

```
red-neuronal-mlp/
├── app.py               # Interfaz gráfica principal
├── README.md            # Este archivo
├── requirements.txt     # Dependencias
├── neuron/              # Módulo de neuronas
│   ├── __init__.py
│   └── neurona.py       # Implementación de la neurona
├── layer/               # Módulo de capas
│   ├── __init__.py
│   └── layer.py         # Implementación de capas
└── neural_network/      # Módulo de red neuronal
    ├── __init__.py
    └── neural_network.py # Implementación de la red MLP
```

## 📚 Documentación Técnica

### Neuron
La clase `Neuron` implementa una neurona individual con:
- Pesos y sesgo aleatorios
- Función de activación sigmoide
- Retropropagación del error

### Layer
La clase `Layer` maneja una capa de neuronas:
- Inicialización de múltiples neuronas
- Propagación hacia adelante
- Retropropagación del error

### NeuralNetwork
La clase `NeuralNetwork` implementa una red MLP completa:
- Añadir múltiples capas
- Entrenamiento con retropropagación
- Predicción con nuevos datos

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor, lee nuestra [guía de contribución](CONTRIBUTING.md) para más detalles.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- A la comunidad de código abierto por su inspiración
- A los desarrolladores de NumPy y Matplotlib por sus increíbles bibliotecas

---

<div align="center">
  Hecho con ❤️ por [Tu Nombre]
</div>
