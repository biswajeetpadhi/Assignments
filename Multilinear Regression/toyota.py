# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 21:51:04 2021

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
toyota = pd.read_csv("E:\\ASSIGNMENT\\Multilinear Regression\\Datasets_MLR\\toyota.csv")

toyota = toyota.iloc[:, :9]

toyota.describe()
toyota.info()
toyota.columns   
toyota.shape  
toyota.head 
toyota.count()    
toyota.isna().sum()       
toyota.isnull().sum()   
      

# Correlation matrix 
corea = toyota.corr()

# preparing model considering all the variables 
         
ml1 = smf.ols('price ~ age + km + hp + cc + doors + gears + tax + weight', data = toyota).fit() # regression model

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
# index 80 is showing high influence so we can exclude that entire row

toyota = toyota.drop(toyota.index[[80]])

# Preparing model                  
ml1 = smf.ols('price ~ age + km + hp + cc + doors + gears + tax + weight', data = toyota).fit() # regression model

# Summary
ml1.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_ag = smf.ols('age ~  km + hp + cc + doors + gears + tax + weight', data = toyota).fit().rsquared  
vif_ag = 1/(1 - rsq_ag) 

rsq_km = smf.ols('km ~  age + hp + cc + doors + gears + tax + weight', data = toyota).fit().rsquared  
vif_km = 1/(1 - rsq_km) 

rsq_hp = smf.ols('hp ~  age + km + cc + doors + gears + tax + weight', data = toyota).fit().rsquared  
vif_hp = 1/(1 - rsq_hp) 

rsq_cc = smf.ols('cc ~  age + km + hp + doors + gears + tax + weight', data = toyota).fit().rsquared  
vif_cc = 1/(1 - rsq_cc) 

rsq_do = smf.ols('doors ~  age + km + hp + cc + gears + tax + weight', data = toyota).fit().rsquared  
vif_do = 1/(1 - rsq_do) 

rsq_ge = smf.ols('gears ~  age + km + hp + cc + doors + tax + weight', data = toyota).fit().rsquared  
vif_ge = 1/(1 - rsq_ge) 

rsq_ta = smf.ols('tax ~  age + km + hp + cc + doors + gears + weight', data = toyota).fit().rsquared  
vif_ta = 1/(1 - rsq_ta) 

rsq_we = smf.ols('weight ~  age + km + hp + cc + doors + gears + tax ', data = toyota).fit().rsquared  
vif_we = 1/(1 - rsq_we) 



# Storing vif values in a data frame
d1 = {'Variables':['age',  'km',    'hp',  'cc',   'doors', 'gears', 'tax', 'weight'],
            'VIF':[ vif_ag, vif_km, vif_hp, vif_cc, vif_do, vif_ge, vif_ta, vif_we]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

#all vif are less than 10

# Prediction
pred = ml1.predict(toyota)

# Q-Q plot
res = ml1.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = toyota.price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
s_train, s_test = train_test_split(toyota, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('price ~ age + km + hp + cc + doors + gears + tax + weight', data = s_train).fit()

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
