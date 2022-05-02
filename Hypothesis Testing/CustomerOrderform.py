# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 16:26:22 2021

@author: biswa
"""


import pandas as pd
import scipy 
from scipy import stats

cusorder=pd.read_csv("E:\\ASSIGNMENT\\Hypothesis Testing\\Datasets_HT\\CustomerOrderform.csv")

cusorder.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cusorder["Phillippines"] = le.fit_transform(cusorder["Phillippines"])
cusorder["Indonesia"] = le.fit_transform(cusorder["Indonesia"])
cusorder["Malta"] = le.fit_transform(cusorder["Malta"])
cusorder["India"] = le.fit_transform(cusorder["India"])

cus = pd.DataFrame([cusorder['Phillippines'].value_counts(),cusorder['Indonesia'].value_counts(),cusorder['Malta'].value_counts(),cusorder['India'].value_counts()])

#H0: Defective Do not Varies by center
#Ha: Defective Varies by center
Chisqres=scipy.stats.chi2_contingency(cus)
Chisqfin=[['','Test Statistic','p-value'],['Sample Data',Chisqres[0],Chisqres[1]]]
Chisqfin

#Hence Defective Do Not Varies by center
