# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 01:21:14 2022

@author: biswa
"""


import pandas as pd
patient = pd.read_csv("E:\\ASSIGNMENT DS\\survival analysis\\Datasets_Survival Analytics\\patient.csv")
patient.head()
patient.describe()


T = patient.followup

# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

kmf.fit(T, event_observed = patient.eventtype)

# Time-line estimations plot 
kmf.plot()
