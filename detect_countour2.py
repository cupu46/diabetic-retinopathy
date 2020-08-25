# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 23:17:46 2020

@author: FADIL
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("2.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap="gray")
plt.show()

edges = cv2.Canny(gray, threshold1=30, threshold2=100)

plt.imshow(edges)
plt.show()