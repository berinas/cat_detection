from sklearn.model_selection import train_test_split
from save_image import saveImage
import os
import numpy as np
from PIL import Image
import random
import cv2

path_cat = 'C:/Users/Berina/Desktop/cat_detection/dataset'
valid_images = [".jpg",".jpeg"]
br = 1
image_list = []

for f in os.listdir(path_cat):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    image = Image.open(os.path.join(path_cat,f))

    image_list.append(image)

random.shuffle(image_list)

train_data ,test_data = train_test_split(image_list, test_size=0.2)

br=1
for img in train_data:
    newpath = 'C:/Users/Berina/Desktop/cat_detection/train'
    img = np.array(img)
    saveImage(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), newpath, str(br)+".jpg")
    br += 1

for img in test_data:
    newpath = 'C:/Users/Berina/Desktop/cat_detection/test'
    img = np.array(img)
    saveImage(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), newpath, str(br)+".jpg")
    br += 1

