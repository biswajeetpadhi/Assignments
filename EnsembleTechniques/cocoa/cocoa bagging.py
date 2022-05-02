# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 11:15:31 2021

@author: biswa
"""



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
predictors = df.loc[:, df.columns!="Company"]
type(predictors)

target = df["Company"]
type(target)


# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)
 

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=42)
from sklearn.ensemble import BaggingClassifier


bag_clf = BaggingClassifier(base_estimator = rf_clf, n_estimators = 50,
                            bootstrap = True, n_jobs = 1, random_state = 42)

bag_clf.fit(x_train, y_train)


from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, bag_clf.predict(x_test))
accuracy_score(y_test, bag_clf.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, bag_clf.predict(x_train))
accuracy_score(y_train, bag_clf.predict(x_train))


######
# GridSearchCV

from sklearn.model_selection import GridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=50, n_jobs=1, random_state=42)

param_grid = {"max_features": [4, 5, 6, 7, 8, 9, 10], "min_samples_split": [2, 3, 10]}

grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(x_train, y_train)

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_


from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, cv_rf_clf_grid.predict(x_test))
accuracy_score(y_test, cv_rf_clf_grid.predict(x_test))

# Evaluation on Training Data
confusion_matrix(y_train, cv_rf_clf_grid.predict(x_train))
accuracy_score(y_train, cv_rf_clf_grid.predict(x_train))
