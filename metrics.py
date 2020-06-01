from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import metrics
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from sklearn.utils.multiclass import unique_labels


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
    model = load_model('modelFinal.h5')
    print('Model loaded.')

    #trainImage, trainLabels = load_data('C:\\Users\\Adam\\PycharmProjects\\CNN\\augmentedColourDS\\Train')
    #testImage, testLabels = load_data('C:\\Users\\Adam\\PycharmProjects\\CNN\\augmentedColourDS\\Test')
    # valImage, valLables = load_data('C:\\Users\\Adam\\PycharmProjects\\CNN\\augmentedColourDS\\Validate')
    #
    # valImage = np.array(valImage)
    # onehot = pd.get_dummies(valLables)
    # valLables = onehot.as_matrix()

    # history = model.evaluate(testImage, testLabels)
    # print(history)

    test_generator = ImageDataGenerator()
    test_data_generator = test_generator.flow_from_directory(
        'C:\\Users\\Adam\\PycharmProjects\\CNN\\augmentedColourDS\\Validate',  # Put your path here
        target_size=(128, 128),
        batch_size=200,
        shuffle=False)

    test_steps_per_epoch = np.math.ceil(test_data_generator.samples / test_data_generator.batch_size)

    predictions = model.predict_generator(test_data_generator, steps=test_steps_per_epoch)
    # Get most likely class
    predicted_classes = np.argmax(predictions, axis=1)

    true_classes = test_data_generator.classes
    class_labels = list(test_data_generator.class_indices.keys())

    report = sklearn.metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)




if __name__ == "__main__":
    main()
