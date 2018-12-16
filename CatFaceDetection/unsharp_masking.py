import cv2
import os
import numpy as np
from PIL import Image
from save_image import saveImage

def unsharpMasking(image):
    gaussian = cv2.GaussianBlur(image, (9,9), 10.0)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0, image) #konacna = image(original)*1.5 - gaussian(zamucena)*0.5 + 0; k=0.5

path = 'C:/Users/Berina/Desktop/cat_detection/uklanjanje_suma'
valid_images = [".jpg",".jpeg"]
br = 1
k = 1

for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    image = np.array(Image.open(os.path.join(path,f)))

    unsharp_image = unsharpMasking(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    newpath = 'C:/Users/Berina/Desktop/cat_detection/maskiranje_neostrina'
    saveImage(unsharp_image, newpath, str(br)+ext.lower())
    br += 1
