# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:31:25 2020

@author: FADIL
"""

import os.path
from os import path
import pandas as pd
import MySQLdb
#db = MySQLdb.connect("localhost","root","","tesis")
#cursor = db.cursor()

#sql = cursor.execute("SELECT count(*) FROM data_gambar")
#sql2 = cursor.fetchone()
#count = sql2[0] 

#x = 1
data = pd.read_csv("E:\\data retina\\labels.csv", sep ="," , encoding='latin-1')

nama_gambar = list(data["Image name"])
RG = list(data["Retinopathy grade"])

#print(nama_gambar)

for x in range(len(nama_gambar)):
    #print(x, nama_gambar[x])
    
    #image = cv2.imread("E://data retina/New folder/Base11/"+nama_gambar[x]+"")

#while x < count+1: 
 #   n = str(x)
    
  #  sqla = cursor.execute("SELECT nama_gambar from data_gambar where id_gambar = '"+n+"'")
   # sqla1 = cursor.fetchone()
    #citra = str(sqla1[0])
    
    cek_path = path.exists("E://data retina/dataset_kaggle_balanced/"+nama_gambar[x]+"")
    
    
    if(cek_path == False):
        print("gambar" ,nama_gambar[x], "=", cek_path)       
        
    x+=1
    