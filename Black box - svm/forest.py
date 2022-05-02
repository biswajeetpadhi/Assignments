# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 17:20:50 2021

@author: biswa
"""

import pandas as pd 
import numpy as np 

forestfires = pd.read_csv("E:\\ASSIGNMENT\\Black box - svm\\Datasets_SVM\\forestfires.csv")

forestfires.head()
forestfires.describe()
forestfires.columns

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train,test = train_test_split(forestfires,test_size = 0.3)

train_X = train.iloc[:,2:30]
train_y = train.iloc[:,30]
test_X  = test.iloc[:,2:30]
test_y  = test.iloc[:,30]


model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) 

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) 

# kernel = rbf # radial base funciton
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) 



