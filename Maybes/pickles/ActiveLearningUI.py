import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from random import sample

class UI:
    def __init__(self, unlabeledData, scatterplot_data):
        self.root = tk.Tk()
        self.U = list(unlabeledData)
        self.L = []
        self.scatterplot_data = scatterplot_data
        self.currentX = None

        self.figure_frame = tk.Frame(master=self.root)

        self.figure = plt.Figure(figsize=(6,5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.figure_frame)
        self.canvas.get_tk_widget().pack()

        self.figure_frame.pack()

        self.nextSample()

        button_frame = tk.Frame()
        
        button1 = tk.Button(master=button_frame, text="Normal", height=10, width=50, command=self.saveNormal)
        button2 = tk.Button(master=button_frame, text="Outlier", height=10, width=50, command=self.saveOutlier)
        button_frame.grid_columnconfigure(0, weight=1)

        button1.grid(row=0, column=1, padx=30, pady=20)
        button2.grid(row=0, column=2, padx=30, pady=20)

        button_frame.pack()

        self.root.mainloop()

    def saveNormal(self):
        self.saveLabel(0)
    
    def saveOutlier(self):
        self.saveLabel(1)

    def saveLabel(self, label):
        self.L.append((self.currentX, label))
        self.nextSample()
    
    def nextSample(self):
        next = sample(self.U, 1)[0]
        self.currentX = next
        self.ax.scatter(self.scatterplot_data[:, 0], self.scatterplot_data[:, 1], color='b')
        self.ax.scatter([next[0]], [next[1]], color='r')
        self.canvas.draw()

    def labelAllRemaining(self):
        pass
scatterdata = np.random.rand(1000, 2)
ui = UI(scatterdata, scatterdata)