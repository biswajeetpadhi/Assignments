# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 21:29:35 2021

@author: biswa
"""

# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values
import statsmodels.formula.api as smf


calcon = pd.read_csv("E:\\ASSIGNMENT\\SimpleLinearRegression\\Datasets_SLR\\calories.csv")

# correlation
calcon.describe()
calcon.info()

np.corrcoef(calcon.weightgain, calcon.caloriescon) 
cov_output = np.cov(calcon.weightgain, calcon.caloriescon)[0, 1]
cov_output
calcon.cov()


# Simple Linear Regression
model = smf.ols('caloriescon ~ weightgain', data = calcon).fit()
model.summary()
pred1 = model.predict(pd.DataFrame(calcon['weightgain']))

# Error calculation
res1 = calcon.caloriescon - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1


######### Model building on Transformed Data
# Log Transformation
# x = log(weightgain); y = caloriescon

np.corrcoef(np.log(calcon.weightgain), calcon.caloriescon) #correlation

model2 = smf.ols('caloriescon ~ np.log(weightgain)', data = calcon).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(calcon['weightgain']))

# Error calculation
res2 = calcon.caloriescon - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = weightgain; y = log(caloriescon)

np.corrcoef(calcon.weightgain, np.log(calcon.caloriescon)) #correlation

model3 = smf.ols('np.log(caloriescon) ~ weightgain', data = calcon).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(calcon['weightgain']))
pred3_at = np.exp(pred3)
pred3_at

# Error calculation
res3 = calcon.caloriescon - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = weightgain; x^2 = weightgain*weightgain; y = log(caloriescon)

model4 = smf.ols('np.log(caloriescon) ~ weightgain + I(weightgain*weightgain)', data = calcon).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(calcon))
pred4_at = np.exp(pred4)
pred4_at

# Error calculation
res4 = calcon.caloriescon - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse
