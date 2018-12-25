import cv2
import os
import numpy as np
from PIL import Image
from save_image import saveImage
from path import start

def denoiseRemoval(image):
    return cv2.medianBlur(image, 3)                     #iskoristen mediana filter za otklanjanje šuma

path = start + 'dataset'
valid_images = [".jpg",".jpeg"]
br = 1

for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    image = Image.open(os.path.join(path,f))
    image=np.array(image)
    processed_image = denoiseRemoval(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    newpath= start + 'uklanjanje_suma'
    saveImage(processed_image, newpath, str(br) + ext.lower())
    br += 1
