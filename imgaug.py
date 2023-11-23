import os
import cv2
import numpy as np

imgdir = "../data/images/train/"
# labeldir = "../labels/"

for root, dirs, files in os.walk(imgdir):
    for name in files:
        img_name = os.path.join(name)
        print(img_name)
        pic = cv2.imread(imgdir + img_name)
        contrast = 1.2
        brightness = 30
        pic_turn = cv2.addWeighted(pic, contrast, pic, 0, brightness)
        cv2.imwrite(imgdir + 'aug_' + img_name, pic_turn)
        # cv2.imshow('original', pic)
        # cv2.imshow('turn', pic_turn)
        cv2.waitKey(0)
