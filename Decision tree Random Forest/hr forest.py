# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 12:33:39 2021

@author: biswa
"""

import pandas as pd

data = pd.read_csv("E:\\ASSIGNMENT\\Decision tree Random Forest\\Datasets_DTRF\\HR_DT.csv")

data.columns
data
data.describe()

data = data.drop(columns=["Position of the employee"])

X=data.iloc[:,:-1].values
Y=data.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.3,random_state=0)

from sklearn.ensemble import RandomForestRegressor as rfg
model = rfg(n_estimators=300,random_state=0)
model.fit(X_train,Y_train)

#Predicting the value
pred=model.predict(X_test)
