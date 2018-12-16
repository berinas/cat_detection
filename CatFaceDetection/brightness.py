import os
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from save_image import saveImage

def changeBrightness(image):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(1.0)

path = 'C:/Users/Berina/Desktop/cat_detection/dataset'
valid_images = [".jpg",".jpeg"]
br = 1

for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    image = Image.open(os.path.join(path,f))

    new_image = changeBrightness(image)
    new_image = np.array(new_image)
    newpath = 'C:/Users/Berina/Desktop/cat_detection/osvjetljenje'
    saveImage(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB), newpath, str(br) + ext.lower())
    br += 1

