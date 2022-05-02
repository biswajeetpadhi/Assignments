# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 22:07:35 2021

@author: biswa
"""

# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import statsmodels.formula.api as smf


deli = pd.read_csv("E:\\ASSIGNMENT\\SimpleLinearRegression\\Datasets_SLR\\deliverytime.csv")

# correlation
deli.describe()
deli.info()

np.corrcoef(deli.sortingtime, deli.deliverytime) 
cov_output = np.cov(deli.sortingtime, deli.deliverytime)[0, 1]
cov_output
deli.cov()


# Simple Linear Regression
model = smf.ols('deliverytime ~ sortingtime', data = deli).fit()
model.summary()
pred1 = model.predict(pd.DataFrame(deli['sortingtime']))

# Error calculation
res1 = deli.deliverytime - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1


######### Model building on Transformed Data
# Log Transformation
# x = log(sortingtime); y = deliverytime

np.corrcoef(np.log(deli.sortingtime), deli.deliverytime) #correlation

model2 = smf.ols('deliverytime ~ np.log(sortingtime)', data = deli).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(deli['sortingtime']))

# Error calculation
res2 = deli.deliverytime - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = sortingtime; y = log(deliverytime)

np.corrcoef(deli.sortingtime, np.log(deli.deliverytime)) #correlation

model3 = smf.ols('np.log(deliverytime) ~ sortingtime', data = deli).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(deli['sortingtime']))
pred3_at = np.exp(pred3)
pred3_at

# Error calculation
res3 = deli.deliverytime - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = sortingtime; x^2 = sortingtime*sortingtime; y = log(deliverytime)

model4 = smf.ols('np.log(deliverytime) ~ sortingtime + I(sortingtime*sortingtime)', data = deli).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(deli))
pred4_at = np.exp(pred4)
pred4_at

# Error calculation
res4 = deli.deliverytime - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse


