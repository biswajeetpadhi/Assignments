# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 21:15:15 2022

@author: biswa
"""



# Import necessary libraries for MLP and reshaping the data structres
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import pandas as pd

# Loading the data set using pandas as data frame format 
concrete = pd.read_csv("E:\\ASSIGNMENT DS\\ann\\Datasets_ANN Assignment/concrete.csv")

predictors = concrete.iloc[:,:8]
target = concrete.iloc[:,[8]]


##Partitioning the data
x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.25)

regr = MLPRegressor(random_state=1, max_iter=500).fit(x_train, y_train)
pred = regr.predict(x_test)
pred
regr.score(x_test, y_test)

