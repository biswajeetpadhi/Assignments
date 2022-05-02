# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 11:10:48 2021

@author: biswa
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv("E:\\ASSIGNMENT\\Decision tree Random Forest\\Datasets_DTRF\\Fraud_check.csv")

data.columns
data
data.describe()

a = list(data['Taxable.Income']) 
plt.boxplot(a) 

data["taxpay"] = pd.cut(data["Taxable.Income"], bins = [0,30000,99619], labels = ["risky", "good"])
 
# Converting into binary
lb = LabelEncoder()
data["Undergrad"] = lb.fit_transform(data["Undergrad"])
data["Marital.Status"] = lb.fit_transform(data["Marital.Status"])
data["Urban"] = lb.fit_transform(data["Urban"])

data = data.drop(columns=["Taxable.Income"])

colnames = list(data.columns)

predictors = colnames[:5]
target = colnames[5]

X = data[predictors]
Y = data[target]

from sklearn.ensemble import RandomForestClassifier as RFC

rfc = RFC(n_jobs=4,oob_score=True,n_estimators=100,criterion="entropy")

np.shape(data) 

rfc.fit(X,Y) # Fitting RandomForestClassifier model from sklearn.ensemble 
rfc.oob_score_

rfc.predict(X)
data['rfc_pred'] = rfc.predict(X)




