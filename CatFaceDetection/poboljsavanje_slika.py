from brightness import changeBrightness
from histogram import histogramEq
import os
from path import start
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from save_image import saveImage

path = start + 'maske'
valid_images = [".jpg", ".jpeg"]
br = 1
path1 = start + 'cropped_masks'


def poboljsaj(img):
    nova = changeBrightness(img)
    return histogramEq(cv2.cvtColor(np.array(nova), cv2.COLOR_BGR2RGB))


for f in os.listdir(path1):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    image = Image.open(os.path.join(path1, f))
    new_image = poboljsaj(image)
    newpath = start + 'poboljsane_slike'
    saveImage(np.array(new_image), newpath, str(br) + ext.lower())
    br += 1
