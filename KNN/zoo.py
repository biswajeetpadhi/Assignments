# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 19:11:38 2021

@author: biswa
"""

import pandas as pd
import numpy as np

zoo_z = pd.read_csv("E:\\ASSIGNMENT\\KNN\\Datasets_KNN\\zoo.csv")

zoo_z.columns

zoo = zoo_z.iloc[:, 1:17] # Excluding type column

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
zoo_n = norm_func(zoo.iloc[:,:])
zoo_n.describe()

X = np.array(zoo_n.iloc[:,:]) # Predictors 
Y = np.array(zoo_z['type']) # Target 


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 21)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

# Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 

