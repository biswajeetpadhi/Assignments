# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 01:40:55 2021

@author: biswa
"""



from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier


import pandas as pd

df = pd.read_csv("E:\\ASSIGNMENT\\EnsembleTechniques\\Datasets_EnsembleTechniques\\Tumor_Ensemble.csv")

# Dummy variables
df.head()
df.info()


# Converting into binary     
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
df["diagnosis"] = lb.fit_transform(df["diagnosis"])

df.head()

# Input and Output Split
X = df.loc[:, df.columns!="diagnosis"]

y = df["diagnosis"]

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],meta_classifier=lr)


for clf, label in zip([clf1, clf2, clf3, sclf], 
                      ['KNN', 
                       'Random Forest', 
                       'Naive Bayes',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))