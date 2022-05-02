# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 16:24:48 2021

@author: biswa
"""


import pandas as pd
import scipy 
from scipy import stats


buyer=pd.read_csv("E:\\ASSIGNMENT\\Hypothesis Testing\\Datasets_HT\\BuyerRatio.csv")
buyer.drop(['Observed Values'],inplace=True,axis = 1)

#H0: Male Female buyer are similar
#Ha: Male Female buyer are not similar
Chisqres = scipy.stats.chi2_contingency(buyer)
Chi_square=[['','Test Statistic','p-value'],['Buyer',Chisqres[0],Chisqres[1]]]
Chi_square

#Hence Male Female buyer are similar
