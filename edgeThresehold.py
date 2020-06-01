'''
Created on 11 Feb 2019

@author: Adam
'''
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2
import math
from os import rename, walk, listdir, makedirs
from os.path import isfile, join, exists, dirname


def getImageSubdir(directoryPath):
    return [f for f in listdir(directoryPath)]


def showImage(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)


def get_images(file_dir):
    return [f for f in listdir(file_dir) if isfile(join(file_dir, f))]


def open_image(file_path):
    image = cv2.imread(file_path, 0)

    #image = cv2.addWeighted(image, 2, image, 0, 80)
    image = cv2.GaussianBlur(image, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    return thresh



def resizeImage(image):
    oldSize = image.shape

    ratio = max(oldSize)

    delta_w = ratio - oldSize[1]
    delta_h = ratio - oldSize[0]
    top, bottom = int(delta_h // 2), int(delta_h - (delta_h // 2))
    left, right = int(delta_w // 2), int(delta_w - (delta_w // 2))

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return new_im


def cropImage(image):
    # applying closing function
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    size = image.shape

    # finding_contours
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0

    for c in cnts:

        rect = cv2.minAreaRect(c)
        w, h = rect[1]

        area = w * h
        if area > max_area:
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            center = rect[0]
            angle = rect[2]
            M = cv2.getRotationMatrix2D(center, angle, 1)
            imgBuff = cv2.warpAffine(image, M, size)

            max_w = int(w)
            max_h = int(h)
            max_area = area

    out = cv2.getRectSubPix(imgBuff, (max_w, max_h), center)
    cv2.drawContours(image, [box], 0, (128, 255, 0), 2)

    return out


def scaleImage(image, upScaleSize):
    croppedSize = image.shape

    ratio = upScaleSize[0]
    newSize = tuple([int(x * ratio) for x in croppedSize])

    delta_w = ratio - croppedSize[1]
    delta_h = ratio - croppedSize[0]
    top, bottom = int(delta_h // 2), int(delta_h - (delta_h // 2))
    left, right = int(delta_w // 2), int(delta_w - (delta_w // 2))

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    new_im = cv2.resize(new_im, (128, 128))

    return new_im


def save_image(array, fname, subDirectory, directory='processed'):
    newPath = join(directory, subDirectory)
    if not exists(newPath):
        makedirs(newPath)

    outputPath = newPath + '/{}'
    io.imsave(outputPath.format(fname), array)


def preprocess(args):
    directoryPath = args
    imageSubdirectories = getImageSubdir(directoryPath)

    for subdirectory in imageSubdirectories:
        i = 0
        subdirectoryPath = join(directoryPath, subdirectory)
        image_files = get_images(subdirectoryPath)

        for file in image_files:
            i += 1
            newFileName = subdirectory + "-" + str(i) + ".jpg"

            image = open_image('{0}\{1}'.format(subdirectoryPath, file))

            resizedImage = resizeImage(image)
            croppedImage = cropImage(resizedImage)
            resizedImage = resizeImage(croppedImage)
            scaledImage = scaleImage(croppedImage, resizedImage.shape)

            save_image(scaledImage, newFileName, subdirectory)


def main():
    preprocess('C:\\Users\\Adam\\Documents\\Assets\\Images\\Leaves\\LeafRead\\field')

if __name__ == "__main__":
    main()