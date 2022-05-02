# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 17:20:50 2021

@author: biswa
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv("E:\\ASSIGNMENT\\Decision tree Random Forest\\Datasets_DTRF\\Company_data.csv")

data.columns
data
data.describe()

a = list(data['Sales']) 
plt.boxplot(a) 

data["sale_s"] = pd.cut(data["Sales"], bins = [-1,7.5,16.27], labels = ["Low", "High"])
 
# Converting into binary
lb = LabelEncoder()
data["ShelveLoc"] = lb.fit_transform(data["ShelveLoc"])
data["Urban"] = lb.fit_transform(data["Urban"])
data["US"] = lb.fit_transform(data["US"])

data = data.drop(columns=["Sales"])

colnames = list(data.columns)

predictors = colnames[:10]
target = colnames[10]

X = data[predictors]
Y = data[target]

#decision tree

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(X,Y)

model.predict(X)
data['dt_pred'] = model.predict(X)



