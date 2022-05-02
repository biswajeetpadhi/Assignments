# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 20:39:51 2022

@author: biswa
"""


import pandas as pd
solar = pd.read_csv("E:\\ASSIGNMENT DS\\forecasting time series\\Datasets_Forecasting/solarpower.csv")

# Pre processing
import numpy as np

solar["t"] = np.arange(1,2559)


# Visualization - Time plot
solar.power.plot()

# Data Partition
Train = solar.head(2528)
Test = solar.tail(30)


####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('power ~ t', data = Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['power']) - np.array(pred_linear))**2))
rmse_linear
