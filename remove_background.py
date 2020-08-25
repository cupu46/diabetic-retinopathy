# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:20:32 2020

@author: FADIL
"""


# Load image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glcm_fitur

# Load image
image_bgr = cv2.imread("E://data retina/dataset_kaggle_balanced/326_left.jpeg")

img = cv2.resize(image_bgr, (650, 550))

# Convert to RGB
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Rectange values: start x, start y, width, height
rectangle = (60, 20, 350, 550)

# Create initial mask
mask = np.zeros(image_rgb.shape[:2], np.uint8)

# Create temporary arrays used by grabCut
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Run grabCut
cv2.grabCut(image_rgb, # Our image
            mask, # The Mask
            rectangle, # Our rectangle
            bgdModel, # Temporary array for background
            fgdModel, # Temporary array for background
            3, # Number of iterations
            cv2.GC_INIT_WITH_RECT) # Initiative using our rectangle

# Create mask where sure and likely backgrounds set to 0, otherwise 1
mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

# Multiply image with new mask to subtract background
image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]



# Show image
#plt.imshow(image_rgb), plt.axis("off")
#plt.show()

image_gray = cv2.cvtColor(image_rgb_nobg, cv2.COLOR_BGR2GRAY)

#_,binary = cv2.threshold(image_gray, 225, 255, cv2.THRESH_BINARY_INV)
    

edges = cv2.Canny(image_gray, threshold1=30, threshold2=100)

plt.imshow(edges)
plt.show()

glcm = glcm_fitur.getGLCM(edges)

glcm2 = glcm_fitur.getGLCM(image_gray)

asm = glcm_fitur.getContrast(glcm)

asm2 = glcm_fitur.getIDM(glcm2)

#print("GLCM canny =",asm)

print("GLCM no Canny =",asm2)
