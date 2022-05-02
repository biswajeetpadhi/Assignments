# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 22:27:56 2021

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
computer = pd.read_csv("E:\\ASSIGNMENT\\Multilinear Regression\\Datasets_MLR\\computer.csv")

computer.describe()
computer.info()
computer.columns   
computer.shape  
computer.head 
computer.count()    
computer.isna().sum()       
computer.isnull().sum()   
      
# Converting into binary     
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
computer["cd"] = lb.fit_transform(computer["cd"])
computer["multi"] = lb.fit_transform(computer["multi"])
computer["premium"] = lb.fit_transform(computer["premium"])

# Correlation matrix 
corea = computer.corr()

# preparing model considering all the variables 
         
ml1 = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend', data = computer).fit() # regression model

# Summary
ml1.summary()

# Q-Q plot
res = ml1.resid
sm.qqplot(res)
plt.show()
 
# Checking whether data has any influential values 
# Influence Index Plots

sm.graphics.influence_plot(ml1)


# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_sp = smf.ols('speed ~  hd + ram + screen + cd + multi + premium + ads + trend', data = computer).fit().rsquared  
vif_sp = 1/(1 - rsq_sp) 

rsq_hd = smf.ols(' hd ~ speed + ram + screen + cd + multi + premium + ads + trend', data = computer).fit().rsquared  
vif_hd = 1/(1 - rsq_hd)

rsq_ra = smf.ols('ram ~ speed + hd + screen + cd + multi + premium + ads + trend', data = computer).fit().rsquared  
vif_ra = 1/(1 - rsq_ra) 

rsq_sc = smf.ols('screen ~ speed + hd + ram + cd + multi + premium + ads + trend', data = computer).fit().rsquared  
vif_sc = 1/(1 - rsq_sc)

rsq_cd = smf.ols('cd ~ speed + hd + ram + screen + multi + premium + ads + trend', data = computer).fit().rsquared  
vif_cd = 1/(1 - rsq_cd)

rsq_mu = smf.ols('multi ~ speed + hd + ram + screen + cd + premium + ads + trend', data = computer).fit().rsquared  
vif_mu = 1/(1 - rsq_mu)

rsq_pr = smf.ols('premium ~ speed + hd + ram + screen + cd + multi + ads + trend', data = computer).fit().rsquared  
vif_pr = 1/(1 - rsq_pr)

rsq_ad = smf.ols('ads ~ speed + hd + ram + screen + cd + multi + premium + trend', data = computer).fit().rsquared  
vif_ad = 1/(1 - rsq_ad)

rsq_tr = smf.ols('trend ~ speed + hd + ram + screen + cd + multi + premium + ads', data = computer).fit().rsquared  
vif_tr = 1/(1 - rsq_tr)


# Storing vif values in a data frame
d1 = {'Variables':['speed', 'hd',   'ram',  'screen', 'cd',   'multi', 'premium', 'ads',  'trend'],
            'VIF':[ vif_sp,  vif_hd, vif_ra, vif_sc,   vif_cd, vif_mu,  vif_pr,    vif_ad, vif_tr  ]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

#all vif are less than 10

# Prediction
pred = ml1.predict(computer)

# Q-Q plot
res = ml1.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = computer.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(ml1)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
s_train, s_test = train_test_split(computer, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('price ~ speed + hd + ram + screen + cd + multi + premium + ads + trend', data = s_train).fit()

# prediction on test data set 
test_pred = model_train.predict(s_test)
test_pred

# test residual values 
test_resid = test_pred - s_test.price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(s_train)
train_pred

# train residual values 
train_resid  = train_pred - s_train.price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
