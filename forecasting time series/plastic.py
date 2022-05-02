# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 20:48:02 2022

@author: biswa
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing # Holt Winter's Exponential Smoothing


plastic = pd.read_csv("E:\\ASSIGNMENT DS\\forecasting time series\\Datasets_Forecasting/plastic.csv")

Train = plastic.head(48)
Test = plastic.tail(12)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["sales"], seasonal = "add", trend = "add", seasonal_periods = 12).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_add_add, Test.sales) 


# Final Model on 100% Data
hwe_model_add_add = ExponentialSmoothing(plastic["sales"], seasonal = "add", trend = "add", seasonal_periods = 12).fit()

# Load the new data which includes the entry for future 4 values
new_data = pd.read_excel("E:\\ASSIGNMENT DS\\forecasting time series\\Datasets_Forecasting/predict plastic.xlsx")

newdata_pred = hwe_model_add_add.predict(start = new_data.index[0], end = new_data.index[-1])
predict_plastic = pd.concat([new_data, newdata_pred], axis = 1)

