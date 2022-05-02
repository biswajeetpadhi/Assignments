# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 16:23:21 2021

@author: biswa
"""


import pandas as pd
import scipy 
from scipy import stats
import statsmodels.api as sm


labtat=pd.read_csv("E:\\ASSIGNMENT\\Hypothesis Testing\\Datasets_HT\\lab_tat_updated.csv")

labtat.info()
print(labtat.isnull().sum())
print(labtat.isna().sum())

#H0 : Follworing normal distribution
#Ha : Not Follworing normal distribution
print(stats.shapiro(labtat.Laboratory_1)) 

print(stats.shapiro(labtat.Laboratory_2)) 

print(stats.shapiro(labtat.Laboratory_3)) 

print(stats.shapiro(labtat.Laboratory_4)) 

#H0 : VAriances are equal
#Ha : VAriances are not equal
scipy.stats.levene(labtat.Laboratory_1,labtat.Laboratory_2,labtat.Laboratory_3,labtat.Laboratory_4)

# H0 : There is difference in turn around time
# Ha : There is no difference in turn around time

from statsmodels.formula.api import ols
labtatmodel=ols('Laboratory_1~Laboratory_2+Laboratory_3+Laboratory_4',data=labtat).fit()
aov_table=sm.stats.anova_lm(labtatmodel,type=2)
print(aov_table)

