import cv2
import os
import numpy as np
from PIL import Image
from skimage.feature import hog
from save_image import saveImage
from resizeimage import resizeimage

def createDescriptor(image):
    image = resizeimage.resize_cover(image, [100, 100])
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4), block_norm='L2', visualize=True)
    return fd, hog_image

hog_images = []
hog_features = []

path = 'C:/Users/Berina/Desktop/cat_detection/poboljsane_slike'
valid_images = [".jpg", ".jpeg"]
br = 1
newpath = 'C:/Users/Berina/Desktop/cat_detection/deskriptor'

for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    image = Image.open(os.path.join(path, f))
    fd, hog_image = createDescriptor(image)
    hog_images.append(hog_image)
    hog_features.append(fd)
    saveImage(hog_image, newpath, str(br) + ext.lower())
    br += 1

