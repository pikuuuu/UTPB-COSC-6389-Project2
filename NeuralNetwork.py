import math
import random
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import csv
import numpy as np

class DataNormalizer:
    @staticmethod
    def min_max_normalize(data):
        """Normalize data to range [0, 1]"""
        data = np.array(data)
        min_vals = data.min(axis=0)
        max_vals = data.max(axis=0)
        
        # Avoid division by zero
        max_vals[max_vals == min_vals] = 1
        
        normalized = (data - min_vals) / (max_vals - min_vals)
        return normalized, min_vals, max_vals

class DataLoader:
    @staticmethod
    def load_csv(filepath, has_header=True, input_columns=None, output_columns=None):
        """
        Load data from CSV file
        
        Args:
            filepath (str): Path to CSV file
            has_header (bool): Whether CSV has a header row
            input_columns (list): Columns to use as inputs (None = auto-detect)
            output_columns (list): Columns to use as outputs (None = last column)
        
        Returns:
            tuple: (inputs, outputs)
        """
        with open(filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            
            # Handle header
            if has_header:
                headers = next(reader)
            
            # Read all data
            data = list(reader)
            data = [[float(cell) for cell in row] for row in data]
            
            # Convert to numpy for easier manipulation
            data = np.array(data)
            
            # Auto-detect input and output columns if not specified
            if input_columns is None:
                input_columns = list(range(data.shape[1] - 1))
            if output_columns is None:
                output_columns = [data.shape[1] - 1]
            
            # Split into inputs and outputs
            inputs = data[:, input_columns]
            outputs = data[:, output_columns]
            
            # Normalize data
            inputs, input_mins, input_maxs = DataNormalizer.min_max_normalize(inputs)
            outputs, output_mins, output_maxs = DataNormalizer.min_max_normalize(outputs)
            
            return inputs, outputs, {
                'input_mins': input_mins, 
                'input_maxs': input_maxs,
                'output_mins': output_mins, 
                'output_maxs': output_maxs
            }


class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)
    
    @staticmethod
    def tanh(x):
        return math.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - x**2
    
    @staticmethod
    def relu(x):
        return max(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return 1 if x > 0 else 0

class Neuron:
    def __init__(self, activation_func, derivative_func, x=0, y=0, input_idx=-1, bias=0.0):
        self.x = x
        self.y = y
        self.inputs = []
        self.outputs = []
        self.index = input_idx
        self.bias = bias
        self.result = 0.0
        self.error = 0.0
        self.activation_func = activation_func
        self.derivative_func = derivative_func

    def connect_input(self, in_n, weight=None):
        if weight is None:
            weight = random.uniform(-1, 1)
        in_axon = Axon(in_n, self, weight)
        self.inputs.append(in_axon)
        return in_axon

    def connect_output(self, out_n):
        out_axon = Axon(self, out_n)
        self.outputs.append(out_axon)
        return out_axon

    def forward_prop(self, inputs, learning_rate=0.1):
        if self.index >= 0:
            self.result = inputs[self.index]
            return self.result

        total = sum(
            in_axon.input.forward_prop(inputs, learning_rate) * in_axon.weight 
            for in_axon in self.inputs
        )
        total += self.bias
        self.result = self.activation_func(total)
        return self.result

    def back_prop(self, learning_rate=0.1):
        gradient = self.derivative_func(self.result)
        delta = self.error * gradient

        for in_axon in self.inputs:
            in_neuron = in_axon.input
            # Update connection weight
            in_axon.weight -= learning_rate * delta * in_neuron.result
            # Propagate error
            in_neuron.error += delta * in_axon.weight

        # Update bias
        self.bias -= learning_rate * delta
        return delta

class Axon:
    def __init__(self, input_neuron, output_neuron, weight=None):
        self.input = input_neuron
        self.output = output_neuron
        self.weight = weight if weight is not None else random.uniform(-1, 1)

class NeuralNetwork:
    def __init__(self, input_count, hidden_layers, output_count, 
                 activation_func=ActivationFunctions.sigmoid, 
                 derivative_func=ActivationFunctions.sigmoid_derivative):
        self.inputs = []
        self.hidden_layers = []
        self.outputs = []
        
        # Create input layer
        for idx in range(input_count):
            self.inputs.append(
                Neuron(activation_func, derivative_func, input_idx=idx, bias=1.0)
            )
        
        # Create hidden layers
        prev_layer = self.inputs
        for layer_width in hidden_layers:
            current_layer = []
            for _ in range(layer_width):
                neuron = Neuron(activation_func, derivative_func)
                # Connect to previous layer
                for prev_neuron in prev_layer:
                    neuron.connect_input(prev_neuron)
                    prev_neuron.connect_output(neuron)
                current_layer.append(neuron)
            self.hidden_layers.append(current_layer)
            prev_layer = current_layer
        
        # Create output layer
        for _ in range(output_count):
            output_neuron = Neuron(activation_func, derivative_func)
            for prev_neuron in prev_layer:
                output_neuron.connect_input(prev_neuron)
                prev_neuron.connect_output(output_neuron)
            self.outputs.append(output_neuron)

    def train(self, inputs, targets, learning_rate=0.1):
        # Forward propagation
        self.predict(inputs)
        
        # Calculate output errors
        for i, output_neuron in enumerate(self.outputs):
            output_neuron.error = targets[i] - output_neuron.result
        
        # Backpropagate
        for output_neuron in reversed(self.outputs):
            output_neuron.back_prop(learning_rate)
        
        # Backpropagate through hidden layers
        for layer in reversed(self.hidden_layers):
            for neuron in layer:
                neuron.back_prop(learning_rate)

    def predict(self, inputs):
        # Reset results for all neurons
        for neuron in self.inputs + [n for layer in self.hidden_layers for n in layer] + self.outputs:
            if neuron.index == -1:
                neuron.result = 0.0
        
        # Perform forward propagation
        return [output_neuron.forward_prop(inputs) for output_neuron in self.outputs]


class EnhancedNeuralNetworkUI:
    def __init__(self, master):
        self.master = master
        master.title("Enhanced Neural Network")
        
        # Create main frame
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configuration Frame
        self.config_frame = tk.Frame(self.main_frame)
        self.config_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Network Configuration Widgets
        self.setup_configuration_widgets()
        
        # Canvas for network visualization
        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, width=800, height=600, bg='white')
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # Status Frame
        self.status_frame = tk.Frame(self.main_frame)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        self.status_label = tk.Label(self.status_frame, text="Ready", anchor='w')
        self.status_label.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # Network state
        self.current_network = None
        self.current_dataset = None
        self.normalization_params = None
    
    def setup_configuration_widgets(self):
        # Input Count
        tk.Label(self.config_frame, text="Number of Inputs:").grid(row=0, column=0)
        self.input_entry = tk.Entry(self.config_frame)
        self.input_entry.grid(row=0, column=1)
        
        # Hidden Layers
        tk.Label(self.config_frame, text="Hidden Layer Sizes:").grid(row=1, column=0)
        self.hidden_layers_entry = tk.Entry(self.config_frame)
        self.hidden_layers_entry.grid(row=1, column=1)
        
        # Output Count
        tk.Label(self.config_frame, text="Number of Outputs:").grid(row=2, column=0)
        self.output_entry = tk.Entry(self.config_frame)
        self.output_entry.grid(row=2, column=1)
        
        # Activation Function
        tk.Label(self.config_frame, text="Activation:").grid(row=3, column=0)
        self.activation_var = tk.StringVar(value="sigmoid")
        activation_options = ["sigmoid", "tanh", "relu"]
        tk.OptionMenu(self.config_frame, self.activation_var, *activation_options).grid(row=3, column=1)
        
        # Buttons
        tk.Button(self.config_frame, text="Load Dataset", command=self.load_dataset).grid(row=4, column=0)
        tk.Button(self.config_frame, text="Create Network", command=self.create_network).grid(row=4, column=1)
        tk.Button(self.config_frame, text="Train Network", command=self.train_network).grid(row=4, column=2)
    
    def load_dataset(self):
        filepath = filedialog.askopenfilename(
            title="Select CSV Dataset", 
            filetypes=[("CSV files", "*.csv")]
        )
        if not filepath:
            return
        
        try:
            # Load dataset
            inputs, outputs, norm_params = DataLoader.load_csv(filepath)
            
            # Update UI with dataset dimensions
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, str(inputs.shape[1]))
            
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, str(outputs.shape[1]))
            
            self.current_dataset = (inputs, outputs)
            self.normalization_params = norm_params
            
            self.status_label.config(text=f"Loaded dataset: {inputs.shape[0]} samples")
            messagebox.showinfo("Dataset Loaded", f"Loaded {inputs.shape[0]} samples with {inputs.shape[1]} inputs and {outputs.shape[1]} outputs")
        
        except Exception as e:
            messagebox.showerror("Loading Error", str(e))
    
    def create_network(self):
        try:
            inputs = int(self.input_entry.get())
            hidden_layers = [int(x) for x in self.hidden_layers_entry.get().split(',')]
            outputs = int(self.output_entry.get())
            
            activation_func_map = {
                "sigmoid": (ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative),
                "tanh": (ActivationFunctions.tanh, ActivationFunctions.tanh_derivative),
                "relu": (ActivationFunctions.relu, ActivationFunctions.relu_derivative)
            }
            
            activation_func, derivative_func = activation_func_map[self.activation_var.get()]
            
            self.current_network = NeuralNetwork(
                inputs, hidden_layers, outputs, 
                activation_func, derivative_func
            )
            
            # Visualize with enhanced details
            self.visualize_network_detailed()
            
            self.status_label.config(text="Network created successfully")
        except Exception as e:
            messagebox.showerror("Network Creation Error", str(e))
    
    def visualize_network_detailed(self):
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        # Collect all layers
        all_layers = [self.current_network.inputs] + \
                     self.current_network.hidden_layers + \
                     [self.current_network.outputs]
        
        # Calculate spacing
        layer_spacing = width / (len(all_layers) + 1)
        max_layer_size = max(len(layer) for layer in all_layers)
        
        # Color gradient for connections
        def get_connection_color(weight):
            # Blue for positive, red for negative weights
            intensity = min(abs(weight) * 5, 255)
            return f'#{int(255 if weight < 0 else 0):02x}{int(255 if weight > 0 else 0):02x}{0:02x}'
        
        # Draw neurons and connections
        for layer_idx, layer in enumerate(all_layers):
            x = layer_spacing * (layer_idx + 1)
            layer_height = height * 0.8
            neuron_spacing = layer_height / (len(layer) + 1)
            
            for neuron_idx, neuron in enumerate(layer):
                y = height * 0.1 + neuron_spacing * (neuron_idx + 1)
                neuron.x, neuron.y = x, y
                
                # Neuron representation
                neuron_color = self.get_neuron_color(neuron.result)
                self.canvas.create_oval(
                    x-15, y-15, x+15, y+15, 
                    fill=neuron_color, 
                    outline='black'
                )
                
                # Connections to next layer
                if layer_idx < len(all_layers) - 1:
                    next_layer = all_layers[layer_idx + 1]
                    for next_neuron in next_layer:
                        # Find corresponding axon
                        matching_axons = [
                            axon for axon in neuron.outputs 
                            if axon.output == next_neuron
                        ]
                        
                        if matching_axons:
                            axon = matching_axons[0]
                            # Connection thickness based on weight magnitude
                            line_width = min(abs(axon.weight) * 3, 5)
                            
                            self.canvas.create_line(
                                x, y, 
                                next_neuron.x, next_neuron.y,
                                fill=get_connection_color(axon.weight),
                                width=line_width
                            )
    
    def get_neuron_color(self, activation):
        """Generate color based on neuron activation"""
        # Blue gradient from pale to deep blue
        intensity = int(min(max(activation, 0), 1) * 255)
        return f'#{intensity:02x}{intensity:02x}ff'
    
    def train_network(self):
        if not self.current_network:
            messagebox.showwarning("Warning", "Create a network first!")
            return
        
        if self.current_dataset is None:
            messagebox.showwarning("Warning", "Load a dataset first!")
            return
        
        inputs, outputs = self.current_dataset
        
        # Training parameters
        epochs = simpledialog.askinteger(
            "Training", "Number of Epochs:", 
            initialvalue=100, minvalue=1
        )
        
        if not epochs:
            return
        
        # Track training progress
        training_loss = []
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Shuffle data
            indices = np.random.permutation(len(inputs))
            
            for idx in indices:
                # Train on single sample
                network_output = self.current_network.predict(inputs[idx])
                self.current_network.train(inputs[idx], outputs[idx])
                
                # Calculate loss (mean squared error)
                epoch_loss = np.mean((network_output - outputs[idx])**2)
                total_loss += epoch_loss
            
            training_loss.append(total_loss / len(inputs))
            
            # Update visualization every 10 epochs
            if epoch % 10 == 0:
                self.visualize_network_detailed()
                self.status_label.config(
                    text=f"Training: Epoch {epoch}, Loss: {training_loss[-1]:.4f}"
                )
                self.master.update()
        
        messagebox.showinfo(
            "Training Complete", 
            f"Training finished in {epochs} epochs\nFinal Loss: {training_loss[-1]:.4f}"
        )

def main():
    root = tk.Tk()
    root.geometry("1200x800")
    app = EnhancedNeuralNetworkUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()