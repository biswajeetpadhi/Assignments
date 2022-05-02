# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 16:21:50 2021

@author: biswa
"""


import pandas as pd
import numpy as np
import scipy 
from scipy import stats

cutlets=pd.read_csv("E:\\ASSIGNMENT\\Hypothesis Testing\\Datasets_HT\\Cutlets.csv")

cutlets.info()
print(cutlets.isnull().sum())
print(cutlets.isna().sum())

from sklearn.impute import SimpleImputer

#mean  imputation
mean_impute=SimpleImputer(missing_values = np.nan,strategy="mean")
cutlets["Unit A"]=pd.DataFrame(mean_impute.fit_transform(cutlets[["Unit A"]]))
cutlets["Unit B"]=pd.DataFrame(mean_impute.fit_transform(cutlets[["Unit B"]]))

#H0 : Follworing normal distribution
#Ha : Not Follworing normal distribution
print(stats.shapiro(cutlets['Unit A'])) 
print(stats.shapiro(cutlets['Unit B'])) 

#H0 : VAriances are equal
#Ha : VAriances are not equal
scipy.stats.levene(cutlets['Unit A'], cutlets['Unit B'])

#H0 : There is significant difference between size of diameter of cutlets
#Ha : There is no significant difference between size of diameter of cutlets
scipy.stats.ttest_ind(cutlets['Unit A'], cutlets['Unit B'])

