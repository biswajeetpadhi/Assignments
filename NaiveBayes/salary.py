# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 20:58:36 2021

@author: biswa
"""


import numpy as np
import pandas as pd

salary_train=pd.read_csv("E:\\ASSIGNMENT\\NaiveBayes\\Datasets_Naive Bayes\\SalaryData_Train.csv")
salary_test=pd.read_csv("E:\\ASSIGNMENT\\NaiveBayes\\Datasets_Naive Bayes\\SalaryData_Test.csv")
salary_train.columns
salary_test.columns
need_encoding=['workclass','education','maritalstatus','occupation','relationship','race','sex','native']

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
for i in need_encoding:
    salary_train[i]=label_encoder.fit_transform(salary_train[i])
    salary_test[i]=label_encoder.fit_transform(salary_test[i])

col_names=list(salary_train.columns)
train_X=salary_train[col_names[0:13]]
train_Y=salary_train[col_names[13]]
test_x=salary_test[col_names[0:13]]
test_y=salary_test[col_names[13]]


#Multinomial Naive Bayes

from sklearn.naive_bayes import MultinomialNB as mn
model=mn()
train_pred=model.fit(train_X,train_Y).predict(train_X)
test_pred=model.fit(train_X,train_Y).predict(test_x)

train_acc=np.mean(train_pred==train_Y)
test_acc=np.mean(test_pred==test_y)
train_acc
test_acc


