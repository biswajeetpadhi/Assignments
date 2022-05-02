# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 01:47:27 2021

@author: biswa
"""



from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier

import pandas as pd

df = pd.read_excel("E:\\ASSIGNMENT\\EnsembleTechniques\\Datasets_EnsembleTechniques\\Coca_Rating_Ensemble.xlsx")

df.head()
df.info()

# Converting into binary     
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
df["Company"] = lb.fit_transform(df["Company"])
df["Name"] = lb.fit_transform(df["Name"])
df["Company_Location"] = lb.fit_transform(df["Company_Location"])
df["Bean_Type"] = lb.fit_transform(df["Bean_Type"])
df["Origin"] = lb.fit_transform(df["Origin"])

df.head()

# Input and Output Split
X = df.loc[:, df.columns!="Company"]

y = df["Company"]

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