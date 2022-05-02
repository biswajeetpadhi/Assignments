# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 10:21:04 2021

@author: biswa
"""

import pandas as pd
import numpy as np

data = pd.read_csv("E:\\ASSIGNMENT\\Decision tree Random Forest\\Datasets_DTRF\\Diabetes.csv")
data.columns
data
data.describe()


colnames = list(data.columns)

predictors = colnames[:8]
target = colnames[8]

X = data[predictors]
Y = data[target]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier as RFC

model = RFC(n_jobs=4,oob_score=True,n_estimators=100,criterion="entropy")

np.shape(data) 

model.fit(X_train,Y_train) # Fitting RandomForestClassifier model from sklearn.ensemble 
model.oob_score_

# Prediction on Test Data
preds = model.predict(X_test)
pd.crosstab(Y_test, preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == Y_test) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(X_train)
pd.crosstab(Y_train, preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == Y_train) # Test Data Accuracy 




