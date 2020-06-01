#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# Support module generated by PAGE version 4.22
#  in conjunction with Tcl version 8.6
#    Apr 03, 2019 10:13:35 AM BST  platform: Windows NT

import sys
import csv
import processImage
import outputConvImages
import tkinter as tk
import numpy as np
from tkinter import filedialog
from PIL import ImageTk, Image
from keras.models import load_model

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

def set_Tk_var():
    global convOutChecked
    convOutChecked = tk.BooleanVar()
    convOutChecked.set(False)


def loadImage():
    global cv2_image
    file_path = tk.filedialog.askopenfilename()

    cv2_image = processImage.processImage(file_path)
    pil_image = Image.fromarray(cv2_image)

    tk_image = ImageTk.PhotoImage(pil_image)

    w.imagePreview.configure(image=tk_image)
    w.imagePreview.image = tk_image
    w.imagePath.configure(text=file_path)

    print('TestGUI_support.loadImage')
    sys.stdout.flush()

def loadModel():
    global model
    model = load_model('modelFinal.h5')
    print('TestGUI_support.loadModel')
    sys.stdout.flush()

def runModel():
    image = np.expand_dims(cv2_image, axis=0)
    class_probs = model.predict(image)
    class_probs = class_probs[0]
    top_values_index = sorted(range(len(class_probs)), key=lambda i: class_probs[i], reverse=True)[:5]

    top5_labels = []
    top5_values = []

    for i in top_values_index:
        top5_labels.append(labels[i])
        top5_values.append(class_probs[i])

    top_results = output_to_string(top5_labels, top5_values)
    w.classOutputs.configure(text=top_results)

    if convOutChecked.get():
        conv0, conv1, conv2, conv3, conv4 = outputConvImages.convOutputs(model, image)

        w.tab0label.configure(image=conv0)
        w.tab0label.image = conv0

        w.tab1label.configure(image=conv1)
        w.tab1label.image = conv1

        w.tab2label.configure(image=conv2)
        w.tab2label.image = conv2

        w.tab3label.configure(image=conv3)
        w.tab3label.image = conv3

        w.tab4label.configure(image=conv4)
        w.tab4label.image = conv4


    print('TestGUI_support.loadModel')
    sys.stdout.flush()


def output_to_string(labels, values):
    top_results = []

    for index in range(0, len(values)):
        probability = values[index]*100
        percentage = round(probability, 4)
        class_result = str(percentage) + '%\n' + labels[index] + '\n'

        top_results.append(class_result)

    return '\n'.join(top_results)

def load_labels():
    csv_path = 'classList.csv'
    csvfile = open(csv_path, 'r')
    csvReader = csv.reader(csvfile, delimiter=",")

    label_list = []
    for row in csvReader:
        label_list.append(row)

    label_list = label_list[0]

    return label_list

def init(top, gui, *args, **kwargs):
    global w, top_level, root
    global labels
    labels = load_labels()
    w = gui
    top_level = top
    root = top

def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None