import cv2
import os
from PIL import Image
from create_descriptor import *
from sklearn import svm
from path import start
from poboljsavanje_slika import poboljsaj
from sklearn.externals import joblib
from resizeimage import resizeimage

trainData = []
labels = []
valid_images = [".jpg", ".jpeg"]

for f in os.listdir(start + 'train'):
    name, ext = os.path.splitext(f)
    if ext.lower() not in valid_images:
        continue
    if 1 <= int(name) <= 20:
        labels.append(1)
    else:
        labels.append(0)
    image = Image.open(os.path.join(start + 'train', f))
    image = resizeimage.resize_cover(image, [100, 100])
    hf, hi = createDescriptor(image)
    trainData.append(hf)

# treniranje modela
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(np.array(trainData), labels)

# save the model to disk
filename = 'model.sav'
joblib.dump(clf, filename)
