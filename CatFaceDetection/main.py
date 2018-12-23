import os
from PIL import Image
import numpy as np
import cv2
from poboljsavanje_slika import *
from create_descriptor import *
from model_training import *
import pickle
from resizeimage import resizeimage
import matplotlib.pyplot as plt

'''
path = 'C:/Users/Berina/Desktop/cat_detection/slike_finalnaValidacija'
loaded_model = joblib.load(open('finalized_model.sav', 'rb'))  # model

def main(path):
    br = 0
    valid_images = [".jpg", ".jpeg"]

    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        image = Image.open(os.path.join(path, f))
        new_image = poboljsaj(image)    # poboljsavanje slika-ujednacavanje histograma+osvjetljenje
        hf, h_image = createDescriptor(image)  # kreiranje deskriptora (HOG)
        
        br += 1


main(path)'''
pics = np.array([
    np.array(Image.open(os.path.join('C:/Users/Berina/Desktop/cat_detection/' + 'slike_finalnaValidacija', f)))
    for f in os.listdir('C:/Users/Berina/Desktop/cat_detection/' + 'slike_finalnaValidacija')])


def sliding_window(img, patch_size=pics[0].shape, istep=2, jstep=2, scale=1.0):
    Ni = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - next(Ni), istep):
        for j in range(0, img.shape[1] - next(Ni), jstep):
            patch = img[i:i + next(Ni), j:j + next(Ni)]
            if scale != 1:
                patch = resizeimage.resize_cover(patch, patch_size)
            yield (i, j), patch


def main(path):
    br = 0
    valid_images = [".jpg", ".jpeg"]

    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        image = Image.open(os.path.join(path, f))
        new_image = poboljsaj(image)  # poboljsavanje slika-ujednacavanje histograma+osvjetljenje
        hf, h_image = createDescriptor(image)  # kreiranje deskriptora (HOG)
        loaded_model = joblib.load(open('finalized_model.sav', 'rb'))  # model

        indices, patches = zip(*sliding_window(new_image))
        patches_hog = np.array([createDescriptor(patch)[0] for patch in patches])

        labels = loaded_model.predict(patches_hog)
        labels.sum()

        fig, ax = plt.subplots()
        ax.imshow(new_image, cmap='gray')
        ax.axis('off')

        Ni, Nj = pics[0].shape
        indices = np.array(indices)

        for i, j in indices[labels == 1]:
            ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',
                                       alpha=0.3, lw=2, facecolor='none'))

        br += 1


main('C:/Users/Berina/Desktop/cat_detection/slike_finalnaValidacija')