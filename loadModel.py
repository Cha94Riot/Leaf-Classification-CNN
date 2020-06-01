import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from keras import models
from keras.models import load_model
from keras.utils import plot_model
import time
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as pil_image
from keras.preprocessing.image import save_img
from keras import layers
from keras.applications import vgg16
from keras import backend as K
from keras import metrics
import os
import pandas as pd


def load_data(data_dir):

    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]

    return directories

def main():
    model = load_model('modelFinal.h5')
    print('Model loaded.')
    #model.summary()

    train_data_dir = 'C:\\Users\\Adam\\PycharmProjects\\CNN\\augmentedColourDS\\Test'
    labels = load_data(train_data_dir)

    image = cv2.imread('C:\\Users\\Adam\\PycharmProjects\\CNN\\augmentedColourDS\\Validate\\celtis_occidentalis\\celtis_occidentalis-10.jpg')
    image = np.expand_dims(image, axis=0)


    layer_outputs = [layer.output for layer in model.layers[:15]]  # Extracts the outputs of the top 12 layers
    activation_model = models.Model(inputs=model.input,
                                    outputs=layer_outputs)  # Creates a model that will return these outputs, given the model input

    activations = activation_model.predict(image)  # Returns a list of five Numpy arrays: one array per layer activation

    layer_names = []
    for layer in model.layers[:12]:
        layer_names.append(layer.name)  # Names of the layers, so you can have them as part of your plot

    images_per_row = 4

    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        n_features = layer_activation.shape[0]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                channel_image -= channel_image.mean()  # Post-processes the feature to make it visually palatable
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')


    # w = 28
    # h = 28
    # fig = plt.figure(figsize=(5, 5))
    # columns = 11
    # rows = 11
    # for i in range(1, columns * rows + 1):
    #     #img = np.random.randint(10, size=(h, w))
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(first_layer_activation[0, :, :, i-1], cmap='gray')
    #
    #
    #
    # plt.show()

    # class_probs = model.predict(image)
    # class_probs = class_probs[0]
    # top_values_index = sorted(range(len(class_probs)), key=lambda i: class_probs[i], reverse=True)[:5]
    # print(top_values_index)
    # for i in top_values_index:
    #     print(labels[i], class_probs[i])

    # root = tk.Tk()
    # root.withdraw()
    # file_path = filedialog.askopenfilename()
    # image = cv2.imread(file_path)
    # image = np.expand_dims(image, axis=0)
    # print(model.predict_classes(image))

if __name__ == "__main__":
    main()
