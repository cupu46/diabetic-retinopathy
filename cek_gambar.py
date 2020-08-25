# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 17:09:01 2020

@author: FADIL
"""
import pandas as pd
import cv2
import numpy as np
import glcm_fitur

data = pd.read_csv("E:\\data retina\labels_kaggle.csv", sep ="," , encoding='latin-1')

nama_gambar = list(data["Image name"])
RG = list(data["Retinopathy grade"])

#print(nama_gambar)
fitur =[]
for x in range(len(nama_gambar)):
    #print(x, nama_gambar[x])
    
    image = cv2.imread("E://data retina/dataset_kaggle_balanced/"+nama_gambar[x]+"")
    
    img = cv2.resize(image, (650, 550))
    
    #_, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# create a binary thresholded image
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
# show it

# find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# draw all contours
    image_contour = cv2.drawContours(gray, contours, -1, (0, 255, 0), 25)
    
    #edges = cv2.Canny(gray, threshold1=30, threshold2=100)

    
    glcm = glcm_fitur.getGLCM(image_contour)
    
    entropi = str(round(glcm_fitur.getEntropy(glcm),4))
    energi = str(round(glcm_fitur.getASM(glcm),4))
    homogenitas = str(round(glcm_fitur.getIDM(glcm),4))
    kontras = str(round(glcm_fitur.getContrast(glcm),4))
    korelasi = str(round(glcm_fitur.getCorrelation(glcm),4))
    
    tbl = nama_gambar[x], entropi, energi, homogenitas, kontras, korelasi, RG[x]
    
    fitur.append(tbl)   
    
    print("gambar",nama_gambar[x],"=", entropi, energi, homogenitas, kontras, korelasi, RG[x])
    
    #print("E://data retina/New folder/Base11/"+nama_gambar[x]+"")
    
df = pd.DataFrame(fitur)
csv = df.to_csv("fitur_kaggle_contour.csv", index= False)