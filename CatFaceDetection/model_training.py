'''import cv2
import os
from PIL import Image

svm = cv2.ml.SVM_create() # Set up SVM for OpenCV 3
svm.setType(cv2.ml.SVM_C_SVC) # Set SVM type
svm.setKernel(cv2.ml.SVM_RBF) # Set SVM Kernel to Radial Basis Function (RBF)
svm.setC(12.5) # Set parameter C
svm.setGamma(0.50625) # Set parameter Gamma

trainData = []
valid_images = [".jpg",".jpeg"]
br = 1

for f in os.listdir('C:/Users/Berina/Desktop/cat_detection/train'):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    image = Image.open(os.path.join('C:/Users/Berina/Desktop/cat_detection/train',f))
    trainData.append(image)
    br += 1'''

