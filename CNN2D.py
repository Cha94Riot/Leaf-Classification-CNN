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

    epochs = 15

    train_data_dir = 'C:\\Users\\Adam\\PycharmProjects\\CNN\\colourProcessedTesting\\Train'
    test_dat_dir = 'C:\\Users\\Adam\\PycharmProjects\\CNN\\colourProcessedTesting\\Test'
    x_train, y_train = load_data(train_data_dir)
    x_test, y_test = load_data(test_dat_dir)

    print("Train Labels: {0}\tTrain Images: {1}".format(len(set(y_train)), len(x_train)))
    print("Test Labels: {0}\tTest Images: {1}".format(len(set(y_test)), len(x_test)))


    x_train = np.array(x_train)
    x_test = np.array(x_test)

    onehot = pd.get_dummies(y_test)
    y_test = onehot.as_matrix()

    onehot = pd.get_dummies(y_train)
    y_train = onehot.as_matrix()

    model = Sequential()

    # Input = 128 x 128 x 3 = 49152
    model.add(Conv2D(8, (3, 3), name='Conv1-1'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # Input = 125 x 125 x 8 = 131072
    model.add(Conv2D(16, (3, 3), strides=(2, 2), name='Conv2-2'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # Input = 61 x 61 x 16 = 59536
    model.add(Conv2D(32, (3, 3), strides=(2, 2), name='Conv3-2'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # Input = 29 x 29 x 32 = 26912
    model.add(Conv2D(64, (3, 3), strides=(2, 2), name='Conv4-2'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # Input = 13 x 13 x 64 = 10816
    model.add(Conv2D(128, (3, 3), strides=(2, 2), name='Conv5-1'))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # Input = 5 x 5 x 128 = 3200
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(256))

    model.add(Dense(160))
    model.add(BatchNormalization())
    model.add(Activation("softmax"))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=200)

    model.save('modelColourDSe25.h5')
    model.save_weights('weightsColourDSe25.h5')

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
