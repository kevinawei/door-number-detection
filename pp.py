import numpy as np
import argparse
import imutils
import time
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def preprocess(img):
    result = []
    img1 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 80, 160, 1)  # turn 60, 120 for the best OCR results
    kernel = np.ones((5, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    edged1 = cv2.Canny(gray, 30, 200)  # np.array(thresh)[1].astype('uint8')
    edged2 = cv2.Canny(mask, 30, 200)
    contours1, _ = cv2.findContours(edged1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    contours2, _ = cv2.findContours(edged2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    contours = contours1 + contours2
    for cnt in contours:
        if cv2.arcLength(cnt, True) > 400:
            approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
               # print("square")
                x, y, w, h = cv2.boundingRect(cnt)
               # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 225), 2)
                newImage = img1[y:y + h, x:x + w]
                result.append((newImage, (x, y, w, h)))

    return result
