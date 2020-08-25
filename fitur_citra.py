# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:02:30 2020

@author: FADIL
"""

import cv2 as cv 
import matplotlib.pyplot as plt
import glcm_fitur 
import MySQLdb
import numpy as np
import pandas as pd 


#db = MySQLdb.connect(host="localhost",user="root",passwd="",db="tesis")
#cursor = db.cursor()

#sql = cursor.execute("select count(*) from data_gambar")
#sql2 = cursor.fetchone()
#count = sql2[0] 

data = pd.read_csv("E:/data retina/labels_kaggle.csv", sep ="," , encoding='latin-1')

nama_gambar = list(data["Image name"])
RG = list(data["Retinopathy grade"])

#a = 1
tbl_citra= []
for x in range(len(data)):
    #n = str(a)
    
    #sqla = cursor.execute("SELECT * from data_gambar where id_gambar = '"+n+"'")
    #sqla1 = cursor.fetchone()
    #citra = str(sqla1[1])
    
    #rd_level = str(sqla1[2])
    #print("E:\data retina\dataset_kaggle\\"+nama_gambar[x]+"")
    image = cv.imread("E:/data retina/dataset_kaggle_balanced/"+nama_gambar[x]+"")
    img = cv.resize(image, (650, 550))
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #_, binary = cv.threshold(gray, 225, 255, cv.THRESH_BINARY_INV)
    
    rectangle = (60, 20, 350, 550)

# Create initial mask
    mask = np.zeros(rgb.shape[:2], np.uint8)
    
    # Create temporary arrays used by grabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Run grabCut
    cv.grabCut(rgb, # Our image
                mask, # The Mask
                rectangle, # Our rectangle
                bgdModel, # Temporary array for background
                fgdModel, # Temporary array for background
                5, # Number of iterations
                cv.GC_INIT_WITH_RECT) # Initiative using our rectangle
    
    # Create mask where sure and likely backgrounds set to 0, otherwise 1
    mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
    
    # Multiply image with new mask to subtract background
    image_rgb = rgb * mask_2[:, :, np.newaxis]

    image_gray = cv.cvtColor(image_rgb, cv.COLOR_BGR2GRAY)
    
    #edges = cv.Canny(gray, threshold1=30, threshold2=100)

    
    glcm = glcm_fitur.getGLCM(image_gray)
    
    entropi = str(round(glcm_fitur.getEntropy(glcm),4))
    energi = str(round(glcm_fitur.getASM(glcm),4))
    homogenitas = str(round(glcm_fitur.getIDM(glcm),4))
    kontras = str(round(glcm_fitur.getContrast(glcm),4))
    korelasi = str(round(glcm_fitur.getCorrelation(glcm),4))
    
    fitur = nama_gambar[x], entropi, energi, homogenitas, kontras, korelasi, RG[x]
    
    tbl_citra.append(fitur)
    
    print("gambar",nama_gambar[x],"=", entropi, energi, homogenitas, kontras, korelasi, RG[x])
    
    #input_citra = 'INSERT INTO `ordo_2`(id,`nama_gambar`, `entropi`, `energi`, `homogenitas`, `kontras`, `korelasi`,rd_level) VALUES ("'+n+'","'+citra+'","'+entropi+'","'+energi+'","'+homogenitas+'","'+kontras+'","'+korelasi+'","'+rd_level+'")'
    #cursor.execute(input_citra)
    #db.commit()
    
    #a+=1

df = pd.DataFrame(tbl_citra)
csv = df.to_csv("fitur_kaggle.csv", index= False)
print("success")
    #print("Entropy = ", glcm_fitur.getEntropy(glcm))
    #print("ASM = ", glcm_fitur.getASM(glcm))
    #print("IDM = ", glcm_fitur.getIDM(glcm))
    #print("Contrast = ", glcm_fitur.getContrast(glcm))
    #print("Correlation =", glcm_fitur.getCorrelation(glcm))