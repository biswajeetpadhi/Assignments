# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 17:22:30 2021

@author: biswa
"""



import pandas as pd
from sklearn.model_selection import train_test_split # train and test 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Importing Data
election = pd.read_csv("E:\\ASSIGNMENT\\Logistic Regression\\Datasets_LR/election.csv", sep = ",")


election.info()
election.isna().sum()
election.isnull().sum()

x = election.iloc[:,1:]
y = election.result

# Create LogisticRegression model
log_model = LogisticRegression()

# Fit our data
log_model.fit(x,y)

# Check our accuracy
log_model.score(x,y)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(x,y)

# Make a new log_model
log_model2 = LogisticRegression()

# Now fit the new model
log_model2.fit(X_train, Y_train)

# Predict the classes of the testing data set
pred = log_model2.predict(X_test)

# Compare the predicted classes to the actual test classes
acc_score = accuracy_score(Y_test,pred)
acc_score

