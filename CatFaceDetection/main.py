import os
from PIL import Image
import numpy as np
import cv2
from poboljsavanje_slika import *
from create_descriptor import *
from model_training import *


path = 'C:/Users/Berina/Desktop/cat_detection/slike_finalnaValidacija'
loaded_model = joblib.load(open('finalized_model.sav', 'rb'))  # model

def main(path):
    br = 0
    valid_images = [".jpg", ".jpeg"]

    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        image = Image.open(os.path.join(path, f))
        new_image = poboljsaj(image)    # poboljsavanje slika-ujednacavanje histograma+osvjetljenje
        hf, h_image = createDescriptor(image)  # kreiranje deskriptora (HOG)

        br += 1


main(path)