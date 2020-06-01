import os
import sys
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Conv3D, MaxPooling2D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization, Activation
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.models import Sequential
from keras.utils import to_categorical

def load_data(data_dir):

    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]

    # Loop through the label directories and collect the data in
    # two lists, labels and images.

    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir)
                      if f.endswith(".jpg")]
        for f in file_names:
            images.append(cv2.imread(f))
            labels.append(d)

    return images, labels

def main():

    epochs = 5

    train_data_dir = 'C:\\Users\\Adam\\PycharmProjects\\CNN\\edgeThreshProcessed'
    images, labels = load_data(train_data_dir)
    print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    onehot = pd.get_dummies(y_test)
    y_test = onehot.as_matrix()

    onehot = pd.get_dummies(y_train)
    y_train = onehot.as_matrix()

    model = Sequential()

    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Dense(161))
    model.add(BatchNormalization())
    model.add(Activation("softmax"))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs)

    model.save('modelAllDataset.h5')


    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'blue', label='Training acc')
    plt.plot(epochs, val_acc, 'red', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'blue', label='Training loss')
    plt.plot(epochs, val_loss, 'red', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
