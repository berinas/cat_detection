import cv2
import os
import numpy as np
from PIL import Image, ImageEnhance
from save_image import saveImage

path = 'C:/Users/Berina/Desktop/cat_detection/poboljsane_slike'
valid_images = [".jpg",".jpeg"]
br = 1
newpath = 'C:/Users/Berina/Desktop/cat_detection/deskriptor'

def createDescriptor(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None) # kp-a list of keypoints ; des-a numpy array of shape Number_of_Keypoints×128.
    image = cv2.drawKeypoints(gray, kp, np.array(image))
    return image

for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    image = Image.open(os.path.join(path,f))
    new_image = createDescriptor(image)
    saveImage(new_image, newpath, str(br) + ext.lower())
    br += 1