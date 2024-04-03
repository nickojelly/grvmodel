import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

# Create a new window
window = tk.Tk()

# Create a figure with two subplots
fig = Figure(figsize=(5, 4), dpi=100)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

# Create a canvas and add the figure to it
canvas = FigureCanvasTkAgg(fig, master=window)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Create a treeview widget
tree = ttk.Treeview(window)

# Define the columns
tree["columns"] = ("dog", "price", "model_output")

# Format the columns
tree.column("#0", width=0, stretch=tk.NO)
tree.column("dog", anchor=tk.W, width=100)
tree.column("price", anchor=tk.W, width=100)
tree.column("model_output", anchor=tk.W, width=100)

# Create the column headings
tree.heading("#0", text="", anchor=tk.W)
tree.heading("dog", text="Dog", anchor=tk.W)
tree.heading("price", text="Price", anchor=tk.W)
tree.heading("model_output", text="Model Output", anchor=tk.W)

# Add the treeview to the window
tree.pack()

# Lists to store the prices and model outputs for each dog
prices = [[] for _ in range(8)]
model_outputs = [[] for _ in range(8)]

# Function to add data to the treeview and update the charts
def add_data():
    # Clear the treeview
    tree.delete(*tree.get_children())

    # Add some example data
    for i in range(8):
        price = random.random()
        model_output = random.random()
        tree.insert(parent='', index='end', iid=i, text="", values=(f"Dog {i+1}", price, model_output))
        prices[i].append(price)
        model_outputs[i].append(model_output)

    # Update the charts
    ax1.clear()
    ax2.clear()
    for i in range(8):
        ax1.plot(prices[i], label=f"Dog {i+1}")
        ax2.plot(model_outputs[i], label=f"Dog {i+1}")
    ax1.legend()
    ax2.legend()
    canvas.draw()

    # Schedule this function to be called again in 1 second
    window.after(1000, add_data)

# Start adding data
add_data()

# Start the Tkinter event loop
window.mainloop()