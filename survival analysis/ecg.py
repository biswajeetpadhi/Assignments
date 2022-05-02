# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 01:27:56 2022

@author: biswa
"""


import pandas as pd
ecg = pd.read_excel("E:\\ASSIGNMENT DS\\survival analysis\\Datasets_Survival Analytics\\ecg.xlsx")
ecg.head()
ecg.describe()

T = ecg.survivaltime

# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

kmf.fit(T, event_observed = ecg.alive)

# Time-line estimations plot 
kmf.plot()
