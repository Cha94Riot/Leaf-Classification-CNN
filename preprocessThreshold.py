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
    #     image_paths = []
    #     for dirpath, dirnames, filenames in walk(file_dir):
    #         for filename in [f for f in filenames if isfile(join(dirpath, f))]:
    #             image_paths = dirpath
    #
    #     return image_paths
    return [f for f in listdir(file_dir) if isfile(join(file_dir, f))]


def open_image(file_path):
    image = cv2.imread(file_path, 0)

    image = cv2.addWeighted(image, 2, image, 0, 80)
    image = cv2.GaussianBlur(image, (7, 7), 0)

    #     showImage(image)
    #
    #     edges = cv2.Canny(image,50,100)
    #
    #     showImage(edges)
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    #    ret, thresh = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)

    #    thresh = cv2.Laplacian(thresh, cv2.CV_8U)

    #     showImage(thresh)

    return thresh


def rotateImage(image):
    image = cv2.GaussianBlur(image, (7, 7), 0)
    thresh = cv2.threshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # applying closing function
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # finding_contours
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rows, cols = image.shape
    maxArea = 0

    for c in cnts:
        area = cv2.contourArea(c)

        if area > maxArea:
            cnt = c
            maxArea = area

    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(image, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

    showImage(image)

    return image


def resizeImage(image):
    oldSize = image.shape

    ratio = max(oldSize)

    #    image = cv2.resize(image, (newSize[1], newSize[0]))

    delta_w = ratio - oldSize[1]
    delta_h = ratio - oldSize[0]
    top, bottom = int(delta_h // 2), int(delta_h - (delta_h // 2))
    left, right = int(delta_w // 2), int(delta_w - (delta_w // 2))

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    #    new_im = cv2.resize(new_im, (128, 128))

    #   showImage(image)

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

            #             x = int(center[0] - w/2)
            #             y = int(center[1] - h/2)

            max_w = int(w)
            max_h = int(h)
            max_area = area

    #     imgBuff = imgBuff[ y:y+max_h, x:x+max_w ]
    #     showImage(image)

    out = cv2.getRectSubPix(imgBuff, (max_w, max_h), center)
    cv2.drawContours(image, [box], 0, (128, 255, 0), 2)

    #      showImage(image)

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

    #       showImage(image)

    return new_im


def save_image(array, fname, directory='processed'):
    newPath = os.path.join(directory, subDirectory)
    if not os.path.exists(newPath):
        os.makedirs(newPath)

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
            #            portraitImage = rotateImage(image)

            resizedImage = resizeImage(image)
            croppedImage = cropImage(resizedImage)
            scaledImage = scaleImage(croppedImage, resizedImage.shape)
            #    reorieant
            #    crop
            #    scale
            #    pad

            save_image(scaledImage, newFileName, subdirectory)


def main():
    #     parser = argparse.ArgumentParser(description='Script to preprocess leaf images')
    #     parser.add_argument('-d', '--file_dir', help='Directory where leaf images stored', required=False)
    #     args = vars(parser.parse_args())
    preprocess('C:\\Users\\Adam\\Documents\\Assets\\Images\\Leaves\\LeafRead\\field')


if __name__ == "__main__":
    main()