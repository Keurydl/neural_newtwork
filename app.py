import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from neural_network.neural_network import NauralNetwork
from layer.layer import Layer

class NeuralNetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Red Neuronal - Interfaz Gráfica")
        self.root.geometry("1000x700")
        
        # Variables
        self.network = NauralNetwork()
        self.training_data = []
        self.training_labels = []
        
        # Crear la interfaz
        self.create_widgets()
        
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo - Configuración
        left_panel = ttk.LabelFrame(main_frame, text="Configuración de la Red", padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Panel derecho - Visualización
        right_panel = ttk.LabelFrame(main_frame, text="Visualización", padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configuración de la red
        ttk.Label(left_panel, text="Tamaño de capa:").pack(anchor=tk.W, pady=(5,0))
        self.layer_size = ttk.Entry(left_panel)
        self.layer_size.pack(fill=tk.X, pady=(0,10))
        
        ttk.Button(left_panel, text="Añadir Capa", command=self.add_layer).pack(fill=tk.X, pady=5)
        
        # Lista de capas
        self.layers_listbox = tk.Listbox(left_panel, height=5)
        self.layers_listbox.pack(fill=tk.X, pady=10)
        
        # Datos de entrenamiento
        ttk.Label(left_panel, text="Datos de Entrada (separados por comas):").pack(anchor=tk.W, pady=(10,0))
        self.input_data = ttk.Entry(left_panel)
        self.input_data.pack(fill=tk.X, pady=(0,5))
        
        ttk.Label(left_panel, text="Etiqueta (separadas por comas):").pack(anchor=tk.W)
        self.label_data = ttk.Entry(left_panel)
        self.label_data.pack(fill=tk.X, pady=(0,10))
        
        ttk.Button(left_panel, text="Añadir Datos", command=self.add_training_data).pack(fill=tk.X, pady=5)
        
        # Parámetros de entrenamiento
        ttk.Label(left_panel, text="Épocas:").pack(anchor=tk.W, pady=(10,0))
        self.epochs = ttk.Entry(left_panel)
        self.epochs.insert(0, "1000")
        self.epochs.pack(fill=tk.X, pady=(0,5))
        
        ttk.Label(left_panel, text="Tasa de Aprendizaje:").pack(anchor=tk.W)
        self.learning_rate = ttk.Entry(left_panel)
        self.learning_rate.insert(0, "0.1")
        self.learning_rate.pack(fill=tk.X, pady=(0,10))
        
        # Botones de control
        ttk.Button(left_panel, text="Entrenar Red", command=self.train_network).pack(fill=tk.X, pady=5)
        ttk.Button(left_panel, text="Probar Red", command=self.test_network).pack(fill=tk.X, pady=5)
        
        # Área de visualización
        self.figure, self.ax = plt.subplots(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Área de resultados
        self.result_text = tk.Text(right_panel, height=10)
        self.result_text.pack(fill=tk.X, pady=(10,0))
        
    def add_layer(self):
        try:
            size = int(self.layer_size.get())
            if size <= 0:
                raise ValueError("El tamaño debe ser mayor que cero")
                
            input_size = 1  # Tamaño predeterminado para la primera capa
            if self.layers_listbox.size() > 0:
                # Para capas ocultas, el tamaño de entrada es el tamaño de la capa anterior
                input_size = int(self.layers_listbox.get(0).split(":")[1])
            
            # Si es la primera capa, pedir el tamaño de entrada
            if self.layers_listbox.size() == 0:
                input_size = int(tk.simpledialog.askstring("Tamaño de entrada", 
                                                         "Ingrese el tamaño de entrada:", 
                                                         parent=self.root))
                
            self.layers_listbox.insert(0, f"Capa {self.layers_listbox.size() + 1}: {size} neuronas (Entrada: {input_size})")
            self.layer_size.delete(0, tk.END)
            
        except ValueError as e:
            messagebox.showerror("Error", f"Por favor ingrese un número válido: {str(e)}")
    
    def add_training_data(self):
        try:
            inputs = [float(x.strip()) for x in self.input_data.get().split(",")]
            labels = [float(x.strip()) for x in self.label_data.get().split(",")]
            
            self.training_data.append(np.array(inputs))
            self.training_labels.append(np.array(labels))
            
            self.input_data.delete(0, tk.END)
            self.label_data.delete(0, tk.END)
            
            messagebox.showinfo("Éxito", f"Datos añadidos. Total: {len(self.training_data)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar los datos: {str(e)}")
    
    def train_network(self):
        if not self.layers_listbox.size():
            messagebox.showerror("Error", "Debe agregar al menos una capa a la red")
            return
            
        if len(self.training_data) == 0:
            messagebox.showerror("Error", "Debe agregar datos de entrenamiento")
            return
            
        try:
            # Crear la red neuronal
            self.network = NauralNetwork()
            
            # Añadir capas
            for i in range(self.layers_listbox.size()):
                layer_info = self.layers_listbox.get(i).split(":")
                neurons = int(layer_info[1].split()[0])
                input_size = int(layer_info[1].split("(")[1].split(" ")[1])
                self.network.add_layer(neurons, input_size)
            
            # Entrenar la red
            X = np.array(self.training_data)
            y = np.array(self.training_labels)
            epochs = int(self.epochs.get())
            lr = float(self.learning_rate.get())
            
            self.network.train(X, y, epochs=epochs, learning_rate=lr)
            
            # Mostrar resultados
            self.plot_training()
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "¡Red entrenada exitosamente!\n\n")
            self.result_text.insert(tk.END, f"Épocas: {epochs}\n")
            self.result_text.insert(tk.END, f"Tasa de aprendizaje: {lr}\n")
            self.result_text.insert(tk.END, f"Pérdida final: {self.network.loss_list[-1]:.6f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar la red: {str(e)}")
    
    def test_network(self):
        if not self.layers_listbox.size():
            messagebox.showerror("Error", "La red no ha sido configurada")
            return
            
        test_input = tk.simpledialog.askstring("Probar Red", 
                                             "Ingrese datos de prueba (separados por comas):",
                                             parent=self.root)
        
        if test_input:
            try:
                inputs = np.array([float(x.strip()) for x in test_input.split(",")])
                prediction = self.network.forward(inputs)
                
                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, f"Entrada: {inputs}\n")
                self.result_text.insert(tk.END, f"Salida predicha: {prediction}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al probar la red: {str(e)}")
    
    def plot_training(self):
        self.ax.clear()
        self.ax.plot(self.network.loss_list, label='Pérdida')
        self.ax.set_title('Curva de Aprendizaje')
        self.ax.set_xlabel('Época')
        self.ax.set_ylabel('Pérdida')
        self.ax.legend()
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = NeuralNetworkApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
