# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 19:46:10 2021

@author: biswa
"""


# Multilinear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # mostly used for visualization purposes 
import statsmodels.formula.api as smf # for regression model
import statsmodels.api as sm
import seaborn as sns
from scipy import stats
import pylab

# loading the data
startup = pd.read_csv("E:\\ASSIGNMENT\\Multilinear Regression\\Datasets_MLR\\startup.csv")

startup = startup.drop(["state"],axis=1)
startup.describe()
startup.info()
startup.columns   
startup.shape  
startup.head 
startup.count()    
startup.isna().sum()       
startup.isnull().sum()   
      
# Correlation matrix 
startup.corr()

# preparing model considering all the variables 
         
ml1 = smf.ols('profit ~ rd + admin + market', data = startup).fit() # regression model

# Summary
ml1.summary()

# Q-Q plot
res = ml1.resid
sm.qqplot(res)
plt.show()
 
# Checking whether data has any influential values 
# Influence Index Plots

sm.graphics.influence_plot(ml1)

# Studentized Residuals = Residual/standard deviation of residuals
# index 49 is showing high influence so we can exclude that entire row

startup = startup.drop(startup.index[[49]])

# Preparing model                  
mln = smf.ols('profit ~ rd + admin + market', data = startup).fit()    

# Summary
mln.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_rd = smf.ols('rd ~ admin + market', data = startup).fit().rsquared  
vif_rd = 1/(1 - rsq_rd) 

rsq_ad = smf.ols('admin ~ rd + market', data = startup).fit().rsquared  
vif_ad = 1/(1 - rsq_ad)

rsq_mk = smf.ols('market ~ admin + rd', data = startup).fit().rsquared  
vif_mk = 1/(1 - rsq_mk) 

# Storing vif values in a data frame
d1 = {'Variables':['rd', 'admin', 'market'], 'VIF':[vif_rd, vif_ad, vif_mk]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

#all vif are less than 10

# Prediction
pred = ml1.predict(startup)

# Q-Q plot
res = ml1.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = startup.profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(ml1)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
s_train, s_test = train_test_split(startup, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('profit ~ rd + admin + market', data = s_train).fit()

# prediction on test data set 
test_pred = model_train.predict(s_test)
test_pred

# test residual values 
test_resid = test_pred - s_test.profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(s_train)
train_pred

# train residual values 
train_resid  = train_pred - s_train.profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
