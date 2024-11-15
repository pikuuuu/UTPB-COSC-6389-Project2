import math
import random
import tkinter as tk
from tkinter import ttk, messagebox
import time


num_inputs = 39
num_hidden_layers = 2
hidden_layer_width = 20
num_outputs = 1
neuron_scale = 5
axon_scale = 1
learning_rate = 0.01
training_data_size = 100
activation_function = "sigmoid"

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1 - x**2

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

class Neuron:
    def __init__(self, x, y, input_idx=-1, bias=0.0):
        self.x = x
        self.y = y
        self.inputs = []
        self.outputs = []
        self.index = input_idx
        self.bias = bias
        self.result = 0.0
        self.error = 0.0
        self.b_prop_done = False

    def connect_input(self, in_n):
        in_axon = Axon(in_n, self)
        self.inputs.append(in_axon)

    def connect_output(self, out_n):
        out_axon = Axon(self, out_n)
        self.outputs.append(out_axon)

    def forward_prop(self, inputs):
        if self.result != 0.0:
            return self.result
        total = 0.0
        if self.index >= 0:
            total = inputs[self.index]
        else:
            for in_axon in self.inputs:
                in_n = in_axon.input
                in_n.forward_prop(inputs)
                in_val = in_n.result * in_axon.weight
                total += in_val
        total += self.bias
        
        if activation_function == "sigmoid":
            self.result = sigmoid(total)
        elif activation_function == "tanh":
            self.result = tanh(total)
        elif activation_function == "relu":
            self.result = relu(total)

    def back_prop(self):
        for out_axon in self.outputs:
            out_n = out_axon.output
            out_n.back_prop()
        if self.b_prop_done:
            return
        
        if activation_function == "sigmoid":
            gradient = sigmoid_derivative(self.result)
        elif activation_function == "tanh":
            gradient = tanh_derivative(self.result)
        elif activation_function == "relu":
            gradient = relu_derivative(self.result)
        
        delta = self.error * gradient
        if self.index == -1:
            for in_axon in self.inputs:
                in_n = in_axon.input
                in_n.error += delta * in_axon.weight
                in_axon.weight -= learning_rate * delta * in_n.result
        self.bias -= learning_rate * delta
        self.b_prop_done = True

    def draw(self, canvas, color='black'):
        canvas.create_oval(self.x - neuron_scale, self.y - neuron_scale, self.x + neuron_scale, self.y + neuron_scale, fill=color)

class Axon:
    def __init__(self, in_n, out_n, weight=None):
        self.input = in_n
        self.output = out_n
        self.weight = weight if weight is not None else random.uniform(-1, 1)

    def draw(self, canvas, color='grey'):
        canvas.create_line(self.input.x, self.input.y, self.output.x, self.output.y, fill=color, width=axon_scale)

class Network:
    def __init__(self):
        self.inputs = []
        self.hidden_layers = []
        self.outputs = []
        
        for idx in range(num_inputs):
            in_n = Neuron(0, 0, idx, 1.0)
            self.inputs.append(in_n)
        
        for layer in range(num_hidden_layers):
            self.hidden_layers.append([])
            for _ in range(hidden_layer_width):
                hidden_n = Neuron(0, 0)
                self.hidden_layers[layer].append(hidden_n)
                if layer == 0:
                    for in_n in self.inputs:
                        hidden_n.connect_input(in_n)
                        in_n.connect_output(hidden_n)
                else:
                    for h_n in self.hidden_layers[layer-1]:
                        hidden_n.connect_input(h_n)
                        h_n.connect_output(hidden_n)
        
        for _ in range(num_outputs):
            out_n = Neuron(0, 0)
            self.outputs.append(out_n)
            for h_n in self.hidden_layers[num_hidden_layers-1]:
                out_n.connect_input(h_n)
                h_n.connect_output(out_n)

    def forward_prop(self, inputs):
        for in_n in self.inputs:
            in_n.result = 0.0
        for h_layer in self.hidden_layers:
            for h_n in h_layer:
                h_n.result = 0.0
        for out_n in self.outputs:
            out_n.result = 0.0
            out_n.forward_prop(inputs)

    def back_prop(self):
        for h_layer in self.hidden_layers:
            for h_n in h_layer:
                h_n.error = 0.0
                h_n.b_prop_done = False
        for out_n in self.outputs:
            out_n.error = 0.0
            out_n.b_prop_done = False
        for in_n in self.inputs:
            in_n.error = 0.0
            in_n.b_prop_done = False
            in_n.back_prop()

    def train(self, data):
        self.forward_prop(data[0])
        for x in range(num_outputs):
            self.outputs[x].error = data[1][x] - self.outputs[x].result
        self.back_prop()

    def test(self, inputs):
        self.forward_prop(inputs)
        return [out_n.result for out_n in self.outputs]

    def draw(self, canvas):
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        
        layer_width = width / (num_hidden_layers + 2)
        
        
        for i in range(num_hidden_layers + 2):
            x = i * layer_width
            canvas.create_line(x, 0, x, height, fill='lightgrey', dash=(4, 2))
        
        for idx, neuron in enumerate(self.inputs):
            neuron.x = layer_width / 2
            neuron.y = height * (idx + 1) / (len(self.inputs) + 1)
            neuron.draw(canvas, 'red')
        
        for layer_idx, layer in enumerate(self.hidden_layers):
            for idx, neuron in enumerate(layer):
                neuron.x = layer_width * (layer_idx + 1.5)
                neuron.y = height * (idx + 1) / (len(layer) + 1)
                neuron.draw(canvas, 'blue')
        
        for idx, neuron in enumerate(self.outputs):
            neuron.x = width - layer_width / 2
            neuron.y = height * (idx + 1) / (len(self.outputs) + 1)
            neuron.draw(canvas, 'green')
        
        for layer in self.hidden_layers + [self.outputs]:
            for neuron in layer:
                for axon in neuron.inputs:
                    axon.draw(canvas)

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Neural Network Visualization")
        self.master.state('zoomed')  

        self.network = None
        self.data = self.generate_dataset()

        self.create_menu()
        self.create_canvas()

    def create_menu(self):
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Generate New Dataset", command=self.generate_new_dataset)
        menubar.add_cascade(label="File", menu=file_menu)

        network_menu = tk.Menu(menubar, tearoff=0)
        network_menu.add_command(label="Generate Network", command=self.generate_network)
        network_menu.add_command(label="Train Network", command=self.train_network)
        menubar.add_cascade(label="Network", menu=network_menu)

        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Network Settings", command=self.open_settings)
        menubar.add_cascade(label="Settings", menu=settings_menu)

    def create_canvas(self):
        self.canvas = tk.Canvas(self.master, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def generate_dataset(self):
        data = []
        for _ in range(1000):
            inputs = [random.uniform(0, 1) for _ in range(num_inputs)]
            output = [sum(inputs) / len(inputs)]  
            data.append((inputs, output))
        return data

    def generate_new_dataset(self):
        self.data = self.generate_dataset()
        messagebox.showinfo("Dataset Generated", "New dataset has been generated.")

    def generate_network(self):
        self.network = Network()
        self.data = self.generate_dataset()  
        self.draw_network()

    def train_network(self):
        if not self.network:
            messagebox.showerror("Error", "Please generate a network first")
            return
        if not self.data:
            messagebox.showerror("Error", "No dataset available")
            return
        
        
        train_window = tk.Toplevel(self.master)
        train_window.title("Training Progress")
        train_window.geometry("400x200")

        
        progress = ttk.Progressbar(train_window, length=300, mode='determinate')
        progress.pack(pady=20)

        
        info_label = tk.Label(train_window, text="Starting training...")
        info_label.pack()
        
        error_label = tk.Label(train_window, text="")
        error_label.pack()

        train_window.update()

        total_error = 0
        for i in range(training_data_size):
            data_point = random.choice(self.data)
            self.network.train(data_point)
            
            
            self.network.forward_prop(data_point[0])
            error = abs(self.network.outputs[0].result - data_point[1][0])
            total_error += error

            
            progress['value'] = (i + 1) / training_data_size * 100
            info_label.config(text=f"Training sample {i+1}/{training_data_size}")
            error_label.config(text=f"Current error: {error:.4f}")
            train_window.update()
            time.sleep(0.01)  

        average_error = total_error / training_data_size
        info_label.config(text="Training complete!")
        error_label.config(text=f"Average error: {average_error:.4f}")
        
        self.draw_network()
        messagebox.showinfo("Training Complete", f"Network trained on {training_data_size} samples\nAverage error: {average_error:.4f}")
        train_window.destroy()

    def open_settings(self):
        settings_window = tk.Toplevel(self.master)
        settings_window.title("Network Settings")

        tk.Label(settings_window, text="Number of Hidden Layers:").grid(row=0, column=0)
        hidden_layers_entry = tk.Entry(settings_window)
        hidden_layers_entry.insert(0, str(num_hidden_layers))
        hidden_layers_entry.grid(row=0, column=1)

        tk.Label(settings_window, text="Hidden Layer Width:").grid(row=1, column=0)
        hidden_width_entry = tk.Entry(settings_window)
        hidden_width_entry.insert(0, str(hidden_layer_width))
        hidden_width_entry.grid(row=1, column=1)

        tk.Label(settings_window, text="Activation Function:").grid(row=2, column=0)
        activation_var = tk.StringVar(value=activation_function)
        tk.OptionMenu(settings_window, activation_var, "sigmoid", "tanh", "relu").grid(row=2, column=1)

        def save_settings():
            global num_hidden_layers, hidden_layer_width, activation_function
            num_hidden_layers = int(hidden_layers_entry.get())
            hidden_layer_width = int(hidden_width_entry.get())
            activation_function = activation_var.get()
            self.network = None  
            settings_window.destroy()

        tk.Button(settings_window, text="Save", command=save_settings).grid(row=3, column=0, columnspan=2)

    def draw_network(self):
        self.canvas.delete("all")
        if self.network:
            self.network.draw(self.canvas)

if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()