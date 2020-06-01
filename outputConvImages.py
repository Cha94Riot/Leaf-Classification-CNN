import numpy as np
from math import sqrt, ceil
from keras import models
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import ImageTk, Image
import cv2

def convOutputs(model, image):

    layer_outputs = [layer.output for layer in model.layers[:15]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(image)

    conv0 = display_activations(activations[2])
    conv1 = display_activations(activations[5])
    conv2 = display_activations(activations[8])
    conv3 = display_activations(activations[11])
    conv4 = display_activations(activations[14])

    return conv0, conv1, conv2, conv3, conv4


def display_activations(layer_activations):
    ratio = 2
    feature_maps = layer_activations.shape[3]
    fig = plt.figure(figsize=(10,5))

    columns = ceil(sqrt(feature_maps*ratio))
    rows = ceil(sqrt(feature_maps*(1/ratio)))

    print(columns)
    print(rows)

    for i in range(1, feature_maps+1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(layer_activations[0, :, :, i-1], cmap='plasma')
        plt.axis('off')

    fig.canvas.draw()  # draw the canvas, cache the renderer

    image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    pil_image = Image.fromarray(image)
    tk_image = ImageTk.PhotoImage(pil_image)

    return tk_image
