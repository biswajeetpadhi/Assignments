# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 10:22:48 2021

@author: biswa
"""


# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import statsmodels.formula.api as smf


gpa = pd.read_csv("E:\\ASSIGNMENT\\SimpleLinearRegression\\Datasets_SLR\\satgpa.csv")

# correlation
gpa.describe()
gpa.info()

np.corrcoef(gpa.satscores, gpa.gpa) 
cov_output = np.cov(gpa.satscores, gpa.gpa)[0, 1]
cov_output
gpa.cov()


# Simple Linear Regression
model = smf.ols('gpa ~ satscores', data = gpa).fit()
model.summary()
pred1 = model.predict(pd.DataFrame(gpa['satscores']))

# Error calculation
res1 = gpa.gpa - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1


######### Model building on Transformed Data
# Log Transformation
# x = log(satscores); y = gpa

np.corrcoef(np.log(gpa.satscores), gpa.gpa) #correlation

model2 = smf.ols('gpa ~ np.log(satscores)', data = gpa).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(gpa['satscores']))

# Error calculation
res2 = gpa.gpa - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = satscores; y = log(gpa)

np.corrcoef(gpa.satscores, np.log(gpa.gpa)) #correlation

model3 = smf.ols('np.log(gpa) ~ satscores', data = gpa).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(gpa['satscores']))
pred3_at = np.exp(pred3)
pred3_at

# Error calculation
res3 = gpa.gpa - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = satscores; x^2 = satscores*satscores; y = log(gpa)

model4 = smf.ols('np.log(gpa) ~ satscores + I(satscores*satscores)', data = gpa).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(gpa))
pred4_at = np.exp(pred4)
pred4_at

# Error calculation
res4 = gpa.gpa - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse


