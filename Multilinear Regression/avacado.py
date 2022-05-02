# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 17:40:28 2021

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
avacado = pd.read_csv("E:\\ASSIGNMENT\\Multilinear Regression\\Datasets_MLR\\avacado.csv")

avacado.describe()
avacado.info()
avacado.columns   
avacado.shape  
avacado.head 
avacado.count()    
avacado.isna().sum()       
avacado.isnull().sum()   
      
# Converting into binary     
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
avacado["type"] = lb.fit_transform(avacado["type"])
avacado["region"] = lb.fit_transform(avacado["region"])

# Correlation matrix 
corea = avacado.corr()

# preparing model considering all the variables 
         
ml1 = smf.ols('price ~ vol + ava1 + ava2 + ava3 + bags + sbags + lbags + xbags + type + year + region', data = avacado).fit() # regression model

# Summary
ml1.summary()

# Q-Q plot
res = ml1.resid
sm.qqplot(res)
plt.show()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_vo = smf.ols('vol ~  ava1 + ava2 + ava3 + bags + sbags + lbags + xbags + type + year + region', data = avacado).fit().rsquared  
vif_vo = 1/(1 - rsq_vo) 

rsq_a1 = smf.ols('ava1 ~  vol +   ava2 + ava3 + bags + sbags + lbags + xbags + type + year + region', data = avacado).fit().rsquared  
vif_a1 = 1/(1 - rsq_a1) 

rsq_a2 = smf.ols('ava2 ~  vol + ava1 +   ava3 + bags + sbags + lbags + xbags + type + year + region', data = avacado).fit().rsquared  
vif_a2 = 1/(1 - rsq_a2) 

rsq_a3 = smf.ols('ava3 ~  vol + ava1 + ava2 +   bags + sbags + lbags + xbags + type + year + region', data = avacado).fit().rsquared  
vif_a3 = 1/(1 - rsq_a3) 

rsq_ba = smf.ols('bags ~  vol + ava1 + ava2 + ava3 +   sbags + lbags + xbags + type + year + region', data = avacado).fit().rsquared  
vif_ba = 1/(1 - rsq_ba) 

rsq_sb = smf.ols('sbags ~  vol + ava1 + ava2 + ava3 + bags +  lbags + xbags + type + year + region', data = avacado).fit().rsquared  
vif_sb = 1/(1 - rsq_sb) 

rsq_lb = smf.ols('lbags ~  vol + ava1 + ava2 + ava3 + bags + sbags +   xbags + type + year + region', data = avacado).fit().rsquared  
vif_lb = 1/(1 - rsq_lb) 

rsq_xb = smf.ols('xbags ~  vol + ava1 + ava2 + ava3 + bags + sbags + lbags +   type + year + region', data = avacado).fit().rsquared  
vif_xb = 1/(1 - rsq_xb) 

rsq_ty = smf.ols('type ~  vol + ava1 + ava2 + ava3 + bags + sbags + lbags + xbags +   year + region', data = avacado).fit().rsquared  
vif_ty = 1/(1 - rsq_ty) 

rsq_ye = smf.ols('year ~  vol + ava1 + ava2 + ava3 + bags + sbags + lbags + xbags + type +  region', data = avacado).fit().rsquared  
vif_ye = 1/(1 - rsq_ye) 

rsq_re = smf.ols('region ~  vol + ava1 + ava2 + ava3 + bags + sbags + lbags + xbags + type + year ', data = avacado).fit().rsquared  
vif_re = 1/(1 - rsq_re) 


# Storing vif values in a data frame
d1 = {'Variables':['vol',  'ava1',    'ava2',  'ava3',   'bags', 'sbags', 'lbags', 'xbags', 'type', 'year', 'region'],
            'VIF':[ vif_vo, vif_a1, vif_a2, vif_a3, vif_ba, vif_sb, vif_lb, vif_xb, vif_ty, vif_ye, vif_re]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

#all vif are less than 10

# Prediction
pred = ml1.predict(avacado)

# Q-Q plot
res = ml1.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = avacado.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
s_train, s_test = train_test_split(avacado, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('price ~ vol + ava1 + ava2 + ava3 + bags + sbags + lbags + xbags + type + year + region', data = s_train).fit()

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
