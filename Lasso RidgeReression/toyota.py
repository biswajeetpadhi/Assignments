# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 12:07:27 2021

@author: biswa
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# loading the data
toyota = pd.read_csv("E:\\ASSIGNMENT\\Lasso RidgeReression\\Datasets_LassoRidge/toyota.csv")

toyota = toyota.iloc[:, :9]
toyota.columns
toyota.info()

# Correlation matrix 
core = toyota.corr()
core

# EDA
eda = toyota.describe()

################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

x = toyota.iloc[:, 1:]
y = toyota.price
lasso.fit(x, y)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(toyota.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(x)

# Adjusted r-square
lasso.score(x, y)

# RMSE
resi_la = pred_lasso - y
np.sqrt(np.mean((resi_la)**2))



### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

x = toyota.iloc[:, 1:]
y = toyota.price

rm.fit(x, y)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(toyota.columns[1:]))

rm.alpha

pred_rm = rm.predict(x)

# Adjusted r-square
rm.score(x, y)

# RMSE
resi_rm = pred_rm - y
np.sqrt(np.mean((resi_rm)**2))

