import os
import cv2
from skimage import io
from sklearn.model_selection import train_test_split

def getImageSubdir(directoryPath):
    return [f for f in os.listdir(directoryPath)]


def extract(args):
    directoryPath = args

    imageSubdirectories = getImageSubdir(directoryPath)
    for subdirectory in imageSubdirectories:

        subdirectoryPath = os.path.join(directoryPath, subdirectory)
        imageNames = getImageSubdir(subdirectoryPath)

        newPath = os.path.join('Test', subdirectory)
        if not os.path.exists(newPath):
            os.makedirs(newPath)

        i = 0

        for i in range(int(len(imageNames)/5)):
            oldPath = os.path.join(subdirectoryPath, imageNames[i])
            outputPath = os.path.join(newPath, imageNames[i])
            os.rename(oldPath, outputPath)


def main():
    extract('C:\\Users\\Adam\\PycharmProjects\\CNN\\altDS')

if __name__ == "__main__":
    main()
