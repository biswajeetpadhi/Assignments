# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 10:57:35 2021

@author: biswa
"""

import numpy as np
import pandas as pd

naive_car= pd.read_csv("E:\\ASSIGNMENT\\NaiveBayes\\Datasets_Naive Bayes\\NB_Car_Ad.csv")

x = naive_car.iloc[:,[2,3]].values
y = naive_car.iloc[:, 4].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB as gnb
model=gnb()
train_pred=model.fit(x_train,y_train).predict(x_train)
test_pred=model.fit(x_train,y_train).predict(x_test)

train_acc=np.mean(train_pred==y_train)
test_acc=np.mean(test_pred==y_test)
train_acc
test_acc
