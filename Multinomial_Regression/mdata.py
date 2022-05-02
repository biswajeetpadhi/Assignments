# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 18:48:49 2021

@author: biswa
"""


### Multinomial Regression ####
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

mdata = pd.read_csv("E:\\ASSIGNMENT\\Multinomial_Regression\\Datasets_Multinomial\\mdata.csv")

mdata.head()
mdata.info()
mdata.describe()
mdata.choice.value_counts()

lb = LabelEncoder()

mdata["sex"] = lb.fit_transform(mdata["sex"])
mdata["ses"] = lb.fit_transform(mdata["ses"])
mdata["schtyp"] = lb.fit_transform(mdata["schtyp"])
mdata["honors"] = lb.fit_transform(mdata["honors"])

# Correlation values between each independent features
mdata.corr()

train, test = train_test_split(mdata, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:, 0])

test_predict = model.predict(test.iloc[:, 1:]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)

train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict) 
