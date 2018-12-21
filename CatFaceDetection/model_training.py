import cv2
import os
from PIL import Image
from create_descriptor import *
from sklearn import svm
from skimage.feature import hog
from sklearn.externals import joblib

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
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(np.array(trainData), labels)

# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(clf, filename)






