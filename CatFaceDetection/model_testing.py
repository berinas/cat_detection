import cv2
import os
from PIL import Image
from create_descriptor import *
from model_training import *
from sklearn import *
from path import start
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

testData = []
labels = []
valid_images = [".jpg", ".jpeg"]

for f in os.listdir(start + 'test'):
    name, ext = os.path.splitext(f)
    if ext.lower() not in valid_images:
        continue
    if 1 <= int(name) <= 20:
        labels.append(1)
    else:
        labels.append(0)
    image = Image.open(os.path.join(start + 'test', f))
    hf, hi = createDescriptor(image)
    testData.append(hf)

y_pred = clf.predict(testData)
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(labels, y_pred)))
conf = metrics.confusion_matrix(labels, y_pred)
print("Confusion matrix:\n%s" % conf)

acc = accuracy_score(y_pred, labels)
print("Accuracy: {}".format(acc))

sensitivity1 = conf[0, 0] / (conf[0, 0] + conf[0, 1])
print('Sensitivity : ', sensitivity1)

specificity1 = conf[1, 1] / (conf[1, 0] + conf[1, 1])
print('Specificity : ', specificity1)
