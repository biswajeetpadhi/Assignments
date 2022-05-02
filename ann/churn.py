# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 21:27:18 2022

@author: biswa
"""



# Import necessary libraries for MLP and reshaping the data structres
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pandas as pd

# Loading the data set using pandas as data frame format 
churn = pd.read_csv("E:\\ASSIGNMENT DS\\ann\\Datasets_ANN Assignment/RPL.csv")
churn = churn.iloc[:,3:]

geo = pd.get_dummies(churn.Geography, dummy_na=False)
gen = pd.get_dummies(churn.Gender, dummy_na=False)
churn = pd.concat([churn,geo,gen], axis= 1)
churn = churn.drop(['Geography', 'Gender'], axis=1)

def norm_func(i):
    x = (i - i.min())/(i.max()-i.min())
    return (x)

churn = norm_func(churn)

predictors = churn.drop(['exited'], axis=1)
target = churn.iloc[:,[8]]

##Partitioning the data
x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size = 0.25)

regr = MLPClassifier(random_state=1, max_iter=500).fit(x_train, y_train)
pred = regr.predict(x_test)
pred
regr.score(x_test, y_test)

