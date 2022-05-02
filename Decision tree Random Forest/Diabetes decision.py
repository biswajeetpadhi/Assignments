# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 01:11:46 2021

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

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

#decision tree

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy


