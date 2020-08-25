# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 22:59:28 2020

@author: FADIL
"""

import cv2 
import matplotlib.pyplot as plt
import glcm_fitur

image = cv2.imread("E://data retina/dataset_kaggle_balanced/326_left.jpeg")

img = cv2.resize(image, (650, 550))
# convert to RGB
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# create a binary thresholded image
_, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
# show it

# find the contours from the thresholded image
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# draw all contours
image_contour = cv2.drawContours(gray, contours, -1, (0, 255, 0), 25)

# show the image with the drawn contours

#canny = cv2.Canny(image_contour, threshold1=30, threshold2=50)

plt.imshow(image_contour)
plt.show()

glcm = glcm_fitur.getGLCM(image_contour)

fitur = glcm_fitur.getIDM(glcm)

print(fitur)