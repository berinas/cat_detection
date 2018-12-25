import os
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from path import start
from save_image import saveImage


def changeBrightness(img):
    return ImageEnhance.Brightness(img).enhance(1.0)


'''
path = start + 'dataset'
valid_images = [".jpg", ".jpeg"]
br = 1

for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    image = Image.open(os.path.join(path, f))
    new_image = np.array(changeBrightness(image))
    newpath = start + 'osvjetljenje'
    saveImage(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB), newpath, str(br) + ext.lower())
    br += 1
    '''
