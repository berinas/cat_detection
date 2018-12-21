import cv2
import os
from PIL import Image
from create_descriptor import *
from sklearn import svm
from skimage.feature import hog

trainData = []
labels = []
valid_images = [".jpg",".jpeg"]

for f in os.listdir('C:/Users/Berina/Desktop/cat_detection/train'):
    name, ext = os.path.splitext(f)
    if ext.lower() not in valid_images:
        continue
    if 1 <= int(name) <= 20:
        labels.append(1)
    else:
        labels.append(0)
    image = Image.open(os.path.join('C:/Users/Berina/Desktop/cat_detection/train',f))
    hf, hi = createDescriptor(image)
    trainData.append(hf)

#treniranje modela
clf = svm.SVC(kernel='poly', C=12.5)
clf.fit(np.array(trainData), labels)