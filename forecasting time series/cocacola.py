# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 20:02:16 2022

@author: biswa
"""


import pandas as pd
cocacola = pd.read_excel("E:\\ASSIGNMENT DS\\forecasting time series\\Datasets_Forecasting/cocacola.xlsx")

cocacola = cocacola.iloc[:,:2]

# Pre processing
import numpy as np

cocacola["t"] = np.arange(1,43)

cocacola["t_square"] = cocacola["t"] * cocacola["t"]
cocacola["log_sales"] = np.log(cocacola["sales"])
cocacola.columns


cocacola['Quarter']= 0

for i in range(42):
    p = cocacola['quarter'][i]
    cocacola['Quarter'][i]= p[0:2]

    
quarter_dummies = pd.DataFrame(pd.get_dummies(cocacola['Quarter']))
cocacola= pd.concat([cocacola, quarter_dummies], axis = 1)

# Visualization - Time plot
cocacola.sales.plot()

# Data Partition
Train = cocacola.head(38)
Test = cocacola.tail(4)


####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('sales ~ t', data = Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['sales']) - np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_sales ~ t', data = Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['sales']) - np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('sales ~ t + t_square', data = Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t", "t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['sales']) - np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('sales ~ Q1+Q2+Q3+Q4', data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3','Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['sales']) - np.array(pred_add_sea))**2))
rmse_add_sea

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_sales ~ Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['sales']) - np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

################## Additive Seasonality Quadratic Trend ############################

add_sea_Quad = smf.ols('sales ~ t+t_square+Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','Q4','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality Linear Trend  ###########

Mul_Add_sea = smf.ols('log_sales ~ t+Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse = pd.DataFrame(data)
table_rmse
