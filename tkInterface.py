import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

def main():
    window = tk.Tk()

    window.title('Leaf Classification')
    window.geometry('900x300')

    def selectImage():
        file_path = tk.filedialog.askopenfilename()

        image = ImageTk.PhotoImage(Image.open(file_path))
        canvas = tk.Label(window, image=image)
        canvas.image(20, 20, image=image)
        canvas.grid(column=2, row=1)

        pathLabel.configure(text=file_path)

    selectImageButton = tk.Button(window, width=10, text="Select Image", command=selectImage)
    pathLabel = tk.Label(window, text='Path to Image')

    runModelButton = tk.Button(window, width=10, text="Run Model")
    convCheckbox = tk.Checkbutton(window, text='Show Convolution Outputs')

    selectImageButton.grid(column=0, row=0, sticky="W")
    pathLabel.grid(column=1, row=0, sticky="W")
    runModelButton.grid(column=0, row=1, sticky="W")
    convCheckbox.grid(column=1, row=1, sticky="W")


    window.mainloop()



if __name__ == "__main__":
    main()
