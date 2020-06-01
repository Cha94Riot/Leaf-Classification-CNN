from keras.models import load_model
import os
import cv2
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.metrics import mean_absolute_percentage_error
from keras import backend as K
import sklearn.metrics

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


    valImage, valLables = load_data('C:\\Users\\Adam\\PycharmProjects\\CNN\\augmentedColourDS\\Validate')
    valImage = np.array(valImage)

    onehot = pd.get_dummies(valLables)
    oneHotValLables = onehot.values

    test_generator = ImageDataGenerator()
    test_data_generator = test_generator.flow_from_directory(
        'C:\\Users\\Adam\\PycharmProjects\\CNN\\augmentedColourDS\\Validate',  # Put your path here
        target_size=(128, 128),
        batch_size=200,
        shuffle=False)
    true_classes = test_data_generator.classes
    class_labels = list(test_data_generator.class_indices.keys())

    print(true_classes)

    predictions = model.predict(valImage)

    predicted_classes = np.argmax(predictions, axis=1)

    for i in range(len(valLables)):
        print(valLables[i]+' '+class_labels[predicted_classes[i]])

    report = sklearn.metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report)

if __name__ == "__main__":
    main()
