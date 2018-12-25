from sklearn.model_selection import train_test_split
from save_image import saveImage
import os
import numpy as np
from PIL import Image
import random
import cv2
from path import start

path_cat = start + 'dataset'
valid_images = [".jpg", ".jpeg"]
image_list = []

for f in os.listdir(path_cat):
    name = os.path.splitext(f)[0]
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    image = Image.open(os.path.join(path_cat, f))
    image_list.append((image, name))

random.shuffle(image_list)

train_data, test_data = train_test_split(image_list, test_size=0.2)

for img in train_data:
    newpath = start + 'train'
    saveImage(cv2.cvtColor(np.array(img[0]), cv2.COLOR_BGR2RGB), newpath, img[1] + ".jpg")

for img in test_data:
    newpath = start + 'test'
    saveImage(cv2.cvtColor(np.array(img[0]), cv2.COLOR_BGR2RGB), newpath, img[1] + ".jpg")
