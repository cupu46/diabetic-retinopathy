# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:01:49 2020

@author: FADIL
"""

# -*- coding: utf-8 -*-


import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import numpy as np 
from sklearn import linear_model, preprocessing 

data = pd.read_csv("ordo_2.csv")
#print(data.head())

prep = preprocessing.LabelEncoder()

nama_gambar = list(data["nama_gambar"])
entropi = list(data["entropi"])
energi = list(data["energi"])
homogenitas  = list(data["homogenitas"])
kontras = list(data["kontras"])
korelasi = list(data["korelasi"])
rd_level = list(data["rd_level"])

#print(entropi)

predict = "rd_level"
#tbl = []
#p = entropi, energi, homogenitas, kontras, korelasi
#print(p)
X = list(zip(entropi, energi, homogenitas, kontras, korelasi))
#X = tbl.append(p)

y = list(rd_level)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)



predicted = model.predict(x_test)

names =["0","1","2","3"]

#print(len(predicted))#

for x in range(len(predicted)):
    print("predicted =", names[predicted[x]], "Actual =", names[y_test[x]])


print(acc)
