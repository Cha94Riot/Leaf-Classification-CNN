import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2
import os

def getImageSubdir(directoryPath):
    return [f for f in os.listdir(directoryPath)]

def load_image(fname):
    return io.imread(fname)


def save_image(array, fname, subDirectory, directory='processed'):
    newPath = os.path.join(directory, subDirectory)
    if not os.path.exists(newPath):
        os.makedirs(newPath)

    outputPath = newPath + '/{}'
    io.imsave(outputPath.format(fname), array)


def padImage(image):
    imageSize = image.shape
    newSize = (max(imageSize))*1.5

    delta_w = newSize - imageSize[1]
    delta_h = newSize - imageSize[0]
    top, bottom = int(delta_h // 2), int(delta_h - (delta_h // 2))
    left, right = int(delta_w // 2), int(delta_w - (delta_w // 2))

    color = [0, 0, 0]
    paddedImage = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return paddedImage


def scale_image(paddedImage):
    return cv2.resize(paddedImage, (256, 256))

def getImages(fileDir):
    return [f for f in os.listdir(fileDir) if os.path.isfile(os.path.join(fileDir, f))]

def openImage(filePath):
    image = cv2.imread(filePath)
    return image

def showImage(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)

def getBoundedBox(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    paddedImage = padImage(image)
    paddedThresh = padImage(thresh)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(paddedThresh, cv2.MORPH_CLOSE, kernel)

    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    maxX = 0
    maxY = 0
    maxW = 0
    maxH = 0

    for c in cnts:

        x, y, w, h = cv2.boundingRect(c)

        area = w * h
        if area > max_area:
            #             x = int(center[0] - w/2)
            #             y = int(center[1] - h/2)
            maxX = x
            maxY = y
            maxW = w
            maxH = h
            maxContour = c
            max_area = area

    if(maxW < maxH):
        maxX = int(maxX - ((maxH - maxW)/2))
        maxW = int(maxH)

    elif(maxW > maxH):
        maxY = int(maxY - ((maxW - maxH) / 2))
        maxH = int(maxW)

    roi = paddedImage[maxY:maxY+maxH, maxX:maxX+maxW]

    return roi

def rotateImage(image, angle, fileName):
    rows, cols, _ = image.shape
    newFileName = fileName + "-" + str(angle) + ".jpg"

    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    return image, newFileName

def preprocess(args):
    directoryPath = args

    imageSubdirectories = getImageSubdir(directoryPath)
    for subdirectory in imageSubdirectories:
        i = 0

        subdirectoryPath = os.path.join(directoryPath, subdirectory)
        image_files = getImages(subdirectoryPath)

        for file in image_files:
            i += 1
            FileName = subdirectory + "-" + str(i)

            image = openImage('{0}\\{1}'.format(subdirectoryPath, file))

            # boundedBox = getBoundedBox(image)
            scaled = scale_image(image)

            image0, newFileName = rotateImage(scaled, 0, FileName)
            save_image(image0, newFileName, subdirectory)

            image90, newFileName = rotateImage(scaled, 90, FileName)
            save_image(image90, newFileName, subdirectory)

            image180, newFileName = rotateImage(scaled, 180, FileName)
            save_image(image180, newFileName, subdirectory)

            image270, newFileName = rotateImage(scaled, 270, FileName)
            save_image(image270, newFileName, subdirectory)


def main():
    preprocess('C:\\Users\\Adam\\Documents\\Assets\\Images\\Leaves\\LeafRead\\field')

if __name__ == "__main__":
    main()
