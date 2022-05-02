# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 14:51:44 2021

@author: biswa
"""

import pandas as pd
import scipy 
from scipy import stats

faltoon=pd.read_csv("E:\\ASSIGNMENT\\Hypothesis Testing\\Datasets_HT\\Fantaloons.csv")

falt = pd.DataFrame([faltoon.Weekdays.value_counts(),faltoon.Weekend.value_counts()])

#H0: Differs by days of week.
#Ha: Do not Differs by days of week.

Chisqres=scipy.stats.chi2_contingency(falt)
Chisqfin=[['','Test Statistic','p-value'],['Sample Data',Chisqres[0],Chisqres[1]]]
Chisqfin

#Do not Differs by days of week.
