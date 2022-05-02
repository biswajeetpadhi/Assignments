# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 09:50:05 2021

@author: biswa
"""


# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import statsmodels.formula.api as smf


emp = pd.read_csv("E:\\ASSIGNMENT\\SimpleLinearRegression\\Datasets_SLR\\emp_data.csv")

# correlation
emp.describe()
emp.info()

np.corrcoef(emp.salaryhike, emp.churnout) 
cov_output = np.cov(emp.salaryhike, emp.churnout)[0, 1]
cov_output
emp.cov()


# Simple Linear Regression
model = smf.ols('churnout ~ salaryhike', data = emp).fit()
model.summary()
pred1 = model.predict(pd.DataFrame(emp['salaryhike']))

# Error calculation
res1 = emp.churnout - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1


######### Model building on Transformed Data
# Log Transformation
# x = log(salaryhike); y = churnout

np.corrcoef(np.log(emp.salaryhike), emp.churnout) #correlation

model2 = smf.ols('churnout ~ np.log(salaryhike)', data = emp).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(emp['salaryhike']))

# Error calculation
res2 = emp.churnout - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = salaryhike; y = log(churnout)

np.corrcoef(emp.salaryhike, np.log(emp.churnout)) #correlation

model3 = smf.ols('np.log(churnout) ~ salaryhike', data = emp).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(emp['salaryhike']))
pred3_at = np.exp(pred3)
pred3_at

# Error calculation
res3 = emp.churnout - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = salaryhike; x^2 = salaryhike*salaryhike; y = log(churnout)

model4 = smf.ols('np.log(churnout) ~ salaryhike + I(salaryhike*salaryhike)', data = emp).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(emp))
pred4_at = np.exp(pred4)
pred4_at

# Error calculation
res4 = emp.churnout - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse


