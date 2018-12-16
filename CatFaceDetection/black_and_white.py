from create_use_save_Mask import saveImage
import os
import numpy as np
from PIL import Image
from save_image import saveImage

def toBlackAndWhite(image):
    return image.convert('L')

path = 'C:/Users/Berina/Desktop/cat_detection/dataset'
valid_images = [".jpg",".jpeg"]
br = 1

for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    image = Image.open(os.path.join(path,f))

    new_image = toBlackAndWhite(image)

    newpath = 'C:/Users/Berina/Desktop/cat_detection/b&w'
    saveImage(np.array(new_image), newpath, str(br) + ext.lower())
    br += 1

