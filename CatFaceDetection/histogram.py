from create_use_save_Mask import saveImage
import os
import numpy as np
from PIL import Image
import cv2
from save_image import saveImage

def histogramEq(image):
    data = image.copy().flatten()
    hist, bins = np.histogram(data, 256, density=True)
    cdf = hist.cumsum()
    cdf = 255 * cdf / cdf[-1]
    img_eq = np.interp(data, bins[:-1], cdf)
    return img_eq.reshape(image.shape)

path = 'C:/Users/Berina/Desktop/cat_detection/dataset'
valid_images = [".jpg",".jpeg"]
br = 1

for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    image = Image.open(os.path.join(path,f))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    new_image = histogramEq(image)

    newpath = 'C:/Users/Berina/Desktop/cat_detection/ujednacavanje_histograma'
    saveImage(np.array(new_image), newpath, str(br) + ext.lower())
    br += 1

