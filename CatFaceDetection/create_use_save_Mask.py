import cv2
import numpy as np
import xlrd
import os
from save_image import saveImage
from PIL import Image


def useMask(img, mask):
    return cv2.bitwise_and(img, mask)

def cropMask(imgName, upLeft, bottomRight, upRight, bottomLeft):
    image = Image.open("C:/Users/Berina/Desktop/cat_detection/klase/cat/" + imgName)
    path1 = 'C:/Users/Berina/Desktop/cat_detection/cropped_masks'

    cropped = image.crop((upLeft[0], upLeft[1], bottomRight[0], bottomRight[1]))
    cropped.save(path1 + "/" + imgName)

def createAndApplyMask(imgName, upLeft, bottomRight):

    img = cv2.imread("C:/Users/Berina/Desktop/cat_detection/klase/cat/" + imgName)  # učitavanja originalne slike
    mask = np.zeros(img.shape, dtype="uint8")                                       # kreiranje crne slike (maske)
    cv2.rectangle(mask, upLeft, bottomRight, (255, 255, 255), -1)                   # crtanje bijelog popunjenog pravougaonika na maski

    maskedImg = useMask(img, mask)                                                  # primijeni masku na sliku

    path = 'C:/Users/Berina/Desktop/cat_detection/maske'
    saveImage(maskedImg, path, imgName)



src = 'C:/Users/Berina/Desktop/cat_detection/anotacije.xlsx'
book = xlrd.open_workbook(src)
work_sheet = book.sheet_by_index(0)
num_rows = work_sheet.nrows
current_row = 1

while current_row < num_rows:

    row_header = work_sheet.cell_value(current_row, 2)
    koordinate = row_header.replace('"', '').replace('{', '').replace('}', '').replace('[', '').replace(']',
                                                                                                        '').replace('x',
                                                                                                                    '').replace(
        'y', '').replace(':', '').split(",")

    upL = (int(koordinate[0]), int(koordinate[1]))
    bottomR = (int(koordinate[4]), int(koordinate[5]))
    upR = (int(koordinate[6]), int(koordinate[7]))
    bottomL = (int(koordinate[2]), int(koordinate[3]))

    imeSlike = work_sheet.cell_value(current_row, 7)
    if str(imeSlike) != "19.jpg" and str(imeSlike) != "20.jpg":
        createAndApplyMask(imeSlike, upL, bottomR)
        cropMask(imeSlike, upL, bottomR, upR, bottomL)
    current_row += 1
