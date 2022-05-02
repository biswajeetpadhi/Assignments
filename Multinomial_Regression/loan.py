# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 19:05:41 2021

@author: biswa
"""


### Multinomial Regression ####
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

loan = pd.read_csv("E:\\ASSIGNMENT\\Multinomial_Regression\\Datasets_Multinomial\\loany.csv")
 
loan.isna().sum()
loan = loan.dropna(axis=1)
loan = loan.iloc[:,:15]

loan.head()
loan.info()
loan.describe()

lb = LabelEncoder()

loan.term = lb.fit_transform(loan.term)
loan.int_rate = lb.fit_transform(loan.int_rate)
loan.grade = lb.fit_transform(loan.grade)
loan.sub_grade = lb.fit_transform(loan.sub_grade)
loan.home_ownership = lb.fit_transform(loan.home_ownership)
loan.verification_status = lb.fit_transform(loan.verification_status)
loan.issue_d = lb.fit_transform(loan.issue_d)

# Correlation values between each independent features
core = loan.corr()

train, test = train_test_split(loan, test_size = 0.2)


# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, :14], train.iloc[:, 14])

test_predict = model.predict(test.iloc[:, :14]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:, 14], test_predict)

train_predict = model.predict(train.iloc[:, :14]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:, 14], train_predict) 
