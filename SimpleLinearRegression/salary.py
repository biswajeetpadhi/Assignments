# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 10:10:06 2021

@author: biswa
"""


# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import statsmodels.formula.api as smf


emp = pd.read_csv("E:\\ASSIGNMENT\\SimpleLinearRegression\\Datasets_SLR\\salary_data.csv")

# correlation
emp.describe()
emp.info()

np.corrcoef(emp.yearsexperience, emp.salary) 
cov_output = np.cov(emp.yearsexperience, emp.salary)[0, 1]
cov_output
emp.cov()


# Simple Linear Regression
model = smf.ols('salary ~ yearsexperience', data = emp).fit()
model.summary()
pred1 = model.predict(pd.DataFrame(emp['yearsexperience']))

# Error calculation
res1 = emp.salary - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1


######### Model building on Transformed Data
# Log Transformation
# x = log(yearsexperience); y = salary

np.corrcoef(np.log(emp.yearsexperience), emp.salary) #correlation

model2 = smf.ols('salary ~ np.log(yearsexperience)', data = emp).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(emp['yearsexperience']))

# Error calculation
res2 = emp.salary - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = yearsexperience; y = log(salary)

np.corrcoef(emp.yearsexperience, np.log(emp.salary)) #correlation

model3 = smf.ols('np.log(salary) ~ yearsexperience', data = emp).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(emp['yearsexperience']))
pred3_at = np.exp(pred3)
pred3_at

# Error calculation
res3 = emp.salary - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = yearsexperience; x^2 = yearsexperience*yearsexperience; y = log(salary)

model4 = smf.ols('np.log(salary) ~ yearsexperience + I(yearsexperience*yearsexperience)', data = emp).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(emp))
pred4_at = np.exp(pred4)
pred4_at

# Error calculation
res4 = emp.salary - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse


