# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv

import Dataset
import knn

path = "ordo_2.csv"
DS = Dataset.Dataset(path)
df = DS.getDF()
print(df.head())
X, Y = DS.getXY()
print(X)
k = 7
#gnb = GNB.GNB(X, Y)
knn = knn.kNN(k, X, Y)
accuracy = knn.getAccuracy()
print(accuracy)