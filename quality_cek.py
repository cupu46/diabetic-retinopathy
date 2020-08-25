# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 19:25:01 2020

@author: FADIL
"""

import pandas as pd
import cv2
import numpy as np
import glcm_fitur
import scipy
from scipy import stats 

data = pd.read_csv("E:\\data retina\\New folder\\Base11\\Annotation_Base11b.csv", sep ="," , encoding='latin-1')

nama_gambar = list(data["Image name"])
RG = list(data["Retinopathy grade"])

tbl = []
for x in range(len(data)):
    #print(x, nama_gambar[x])
    
    image = cv2.imread("E://data retina/New folder/Base11/"+nama_gambar[x]+"")


    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)

#snr = scipy.stats.signaltonoise(image, axis=None)

#glcm = glcm_fitur.getGLCM(gray)
#mean = glcm_fitur.getMean(glcm)
#stdev = glcm_fitur.getStandardDeviation(glcm)

    mean = np.mean(gray)
    stdev = np.std(gray)
    SNR = mean/stdev
    
    cek = nama_gambar[x], SNR
    
    tbl.append(cek) 
    
print(tbl) 
df = pd.DataFrame(tbl)
csv = df.to_csv("cek_gambar.csv", index= False)



