from brightness import changeBrightness
from histogram import histogramEq
import os
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from save_image import saveImage
import xlrd

path = 'C:/Users/Berina/Desktop/cat_detection/maske'
valid_images = [".jpg",".jpeg"]
br = 1
path1 = 'C:/Users/Berina/Desktop/cat_detection/cropped_masks'

def poboljsaj(image):
    new_image = changeBrightness(image)
    new_image = histogramEq(cv2.cvtColor(np.array(new_image), cv2.COLOR_BGR2RGB))
    return new_image

for f in os.listdir(path1):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    image = Image.open(os.path.join(path1,f))
    new_image = poboljsaj(image)
    newpath = 'C:/Users/Berina/Desktop/cat_detection/poboljsane_slike'
    saveImage(np.array(new_image), newpath, str(br) + ext.lower())
    br += 1
