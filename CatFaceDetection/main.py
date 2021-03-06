from model_training import *
import matplotlib.pyplot as plt
import time
import cv2
import os
import numpy as np
from PIL import Image
from skimage.feature import hog
from path import start
import imutils
from resizeimage import resizeimage

def pyramid(image, scale=1.5, minSize=(30, 30)):
    yield image

    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        yield image


def sliding_window(image, stepSize=25, windowSize=(100, 100)):
    for y in range(0, image.shape[0] - windowSize[0], stepSize):
        for x in range(0, image.shape[1] - windowSize[1], stepSize):
            patch = image[y:y + windowSize[0], x:x + windowSize[1]]
            yield (x, y), patch


def show_slider(image):
    for resized in pyramid(image):
        for (x, y), window in sliding_window(resized, 50):
            if window.shape[0] != 100 or window.shape[1] != 100:
                continue

            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + 100, y + 100), (0, 255, 0), 3)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            time.sleep(0.025)


def main(path):
    valid_images = [".jpg", ".jpeg"]

    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        image = Image.open(os.path.join(path, f))
        image = resizeimage.resize_cover(image, [200, 200])
        #new_image = poboljsaj(image)
        new_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        hf, h_image = createDescriptor(image)  # kreiranje deskriptora (HOG)
        loaded_model = joblib.load(open('model.sav', 'rb'))  # model

        indices, patches = zip(*sliding_window(new_image))
        show_slider(new_image)

        imageForSliding = Image.open(os.path.join(path, f))
        imageForSliding = cv2.cvtColor(np.array(imageForSliding), cv2.COLOR_BGR2GRAY)
        show_slider(imageForSliding)

        patches_hog = np.array([
            hog(patch, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4), block_norm='L2',
                visualize=True)[0]
            for patch in patches])

        labels = loaded_model.predict(patches_hog)

        fig, ax = plt.subplots()
        ax.imshow(new_image, cmap='gray')
        ax.axis('off')

        indices = np.array(indices)

        clone = new_image.copy()
        upperLeftX = 0
        upperleftY = 0
        lowerRightX = 0
        lowerRightY = 0
        first = False
        for i, j in indices[labels == 1]:
            if not first:
                upperLeftX = i
                upperLeftY = j
                first = True
            else:
                lowerRightX = i
                lowerRightY = j
        if first:
            cv2.rectangle(clone, (upperLeftX, upperLeftY + 100), (lowerRightX + 100, lowerRightY), (0, 255, 0), 3)
            cv2.rectangle(clone, (upperLeftX, upperLeftY + 120), (lowerRightX + 100, lowerRightY + 100), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(clone, "cat", (lowerRightX, lowerRightY + 115), font, 0.5, (255, 255, 255), 1)
        saveImage(clone, start+'result',f)
        cv2.imshow("Detection", clone)
        cv2.waitKey(1)


main(start + 'slike_finalnaValidacija')
