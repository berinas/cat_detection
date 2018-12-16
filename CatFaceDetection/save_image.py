import cv2
import os
from PIL import Image

def saveImage(img, path, name):
    cv2.imwrite(os.path.join(path, name), img)