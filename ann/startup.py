# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 16:50:27 2022

@author: biswa
"""

# Import necessary libraries for MLP and reshaping the data structres
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import pandas as pd

# Loading the data set using pandas as data frame format 
startup = pd.read_csv("E:\\ASSIGNMENT DS\\ann\\Datasets_ANN Assignment/50startup.csv")

startup = startup.iloc[:,:4]

def norm_func(i):
    x = (i - i.min())/(i.max()-i.min())
    return (x)

predictors = startup.iloc[:,[1,2,3]]

target = startup.iloc[:,[0]]

##Partitioning the data
x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.25)

x_train = norm_func(x_train)
y_train = norm_func(x_train)

regr = MLPRegressor(random_state=1, max_iter=500).fit(x_train, y_train)
pred = regr.predict(x_test)
pred
