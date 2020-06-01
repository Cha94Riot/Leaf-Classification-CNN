from skimage import io
import cv2
import os

def getImageSubdir(directoryPath):
    return [f for f in os.listdir(directoryPath)]

def getImages(fileDir):
    return [f for f in os.listdir(fileDir) if os.path.isfile(os.path.join(fileDir, f))]

def openImage(filePath):
    image = cv2.imread(filePath)
    return image

def rotateImage(image, angle, fileName):
    rows, cols, _ = image.shape
    newFileName = fileName + "-" + str(angle) + ".jpg"

    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    return image, newFileName

def save_image(array, fname, subDirectory, directory='augmentedDS'):
    newPath = os.path.join(directory, subDirectory)
    if not os.path.exists(newPath):
        os.makedirs(newPath)

    outputPath = newPath + '/{}'
    io.imsave(outputPath.format(fname), array)

def preprocess(args):
    directoryPath = args

    imageSubdirectories = getImageSubdir(directoryPath)
    for subdirectory in imageSubdirectories:
        subdirectoryPath = os.path.join(directoryPath, subdirectory)
        image_files = getImages(subdirectoryPath)

        for file in image_files:
            #newFileName = subdirectory + "-" + str(i) + ".jpg"
            image = openImage('{0}\\{1}'.format(subdirectoryPath, file))

            save_image(image, file, subdirectory)

            image90, newFileName = rotateImage(image, 90, file)
            save_image(image90, newFileName, subdirectory)

            image180, newFileName = rotateImage(image, 180, file)
            save_image(image180, newFileName, subdirectory)

            image270, newFileName = rotateImage(image, 270, file)
            save_image(image270, newFileName, subdirectory)

def main():
    preprocess('C:\\Users\\Adam\\PycharmProjects\\CNN\\colourProcessedBackup')

if __name__ == "__main__":
    main()
