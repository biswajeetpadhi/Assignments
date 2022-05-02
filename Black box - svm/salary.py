# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 22:37:47 2021

@author: biswa
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Import Data
salary_data_test = pd.read_csv('E:\\ASSIGNMENT\\Black box - svm\\Datasets_SVM/Salary_test.csv')
salary_data_train = pd.read_csv('E:\\ASSIGNMENT\\Black box - svm\\Datasets_SVM/Salary_train.csv')


str_le = ["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in str_le:
    salary_data_train[i]= le.fit_transform(salary_data_train[i])
    salary_data_test[i]=  le.fit_transform(salary_data_test[i])
    

mapping = {' >50K': 1, ' <=50K': 2}
salary_data_test = salary_data_test.replace({'Salary': mapping})
salary_data_train = salary_data_train.replace({'Salary': mapping})

X_train = salary_data_train.drop(labels='Salary',axis=1)
y_train = salary_data_train['Salary']
X_test = salary_data_test.drop('Salary',axis=1)
y_test= salary_data_test['Salary']

model_linear = SVC(kernel='linear',)
model_linear.fit(X_train,y_train)
y_pred_linear = model_linear.predict(X_test)
accuracy_score(y_test,y_pred_linear)

model_rbf=SVC(kernel='rbf',gamma=1)
model_rbf.fit(X_train,y_train)
y_pred_rbf = model_rbf.predict(X_test)
accuracy_score(y_test,y_pred_rbf)

model_poly=SVC(kernel='poly')
model_poly.fit(X_train,y_train)
y_pred_poly = model_poly.predict(X_test)
accuracy_score(y_test,y_pred_poly)

model_sig=SVC(kernel='sigmoid')
model_sig.fit(X_train,y_train)
y_pred_sig = model_sig.predict(X_test)
accuracy_score(y_test,y_pred_sig)