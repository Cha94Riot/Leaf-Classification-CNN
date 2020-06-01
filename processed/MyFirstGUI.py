import cv2
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image,ImageTk

class App:
    def __init__(self, master):

        frame = tk.Frame(master)

        self.selectImageButton = tk.Button(frame, width=10, text="Select Image", command=self.selectImage)
        self.selectImageButton.grid(column=0, row=0, sticky="W")

        self.pathLabel = tk.Label(frame, text='Path to Image')
        self.pathLabel.grid(column=1, row=0, sticky="W")

        self.runModelButton = tk.Button(frame, width=10, text="Run Model", command=self.increase)
        self.runModelButton.grid(column=0, row=1, sticky="W")

        self.convCheckbox = tk.Checkbutton(frame, text='Show Convolution Outputs')
        self.convCheckbox.grid(column=1, row=1, sticky="W")

        self.imagePlace = tk.Label(frame)
        self.imagePlace.grid(column=2, row=1)

        fig = Figure()
        ax = fig.add_subplot(111)
        self.line, = ax.plot(range(10))

        self.canvas = FigureCanvasTkAgg(fig, master=master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)
        frame.pack(side='left')

    def selectImage(self):
        file_path = tk.filedialog.askopenfilename()
        image = ImageTk.PhotoImage(file=file_path)

        self.imagePlace.image = image
        self.imagePlace.configure(image=image)
        self.pathLabel.configure(text=file_path)

    def increase(self):
        x, y = self.line.get_data()
        self.line.set_ydata(y + 0.2 * x)
        self.canvas.draw()

root = tk.Tk()
app = App(root)
root.mainloop()