# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 14:44:41 2022

@author: biswa
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import t
import numpy as np

cars = pd.read_csv('E:\\ASSIGNMENT DS\\Statistics\\probability dataset/Cars.csv')
adipose = pd.read_csv('E:\\ASSIGNMENT DS\\Statistics\\probability dataset/wc-at.csv')

cars.columns
adipose.columns

#Q1
# P(MPG>38)
1-norm.cdf(38,cars.MPG.mean(),cars.MPG.std())

# P(MPG<40)
norm.cdf(40,cars.MPG.mean(),cars.MPG.std())

# P (20<PG<50)
norm.cdf(0.50,cars.MPG.mean(),cars.MPG.std())-norm.cdf(0.20,cars.MPG.mean(),cars.MPG.std()) 


#Q2
#a)	Check whether the MPG of Cars follows Normal Distribution
sns.boxplot(cars.MPG)
plt.hist(cars.MPG)

#b)	Check Whether the Adipose Tissue (AT) and Waist Circumference (Waist) from wc-at data set follows Normal Distribution
sns.boxplot(adipose.Waist)
plt.hist(adipose.Waist)

sns.boxplot(adipose.AT)
plt.hist(adipose.AT)

#Q3
# Z-score of 90% confidence interval 
norm.ppf(0.90)

# Z-score of 94% confidence interval
norm.ppf(0.94)

# Z-score of 60% confidence interval
norm.ppf(0.60)

#Q4
# t scores of 95% confidence interval for sample size of 25
t.ppf(0.95,24)  # df = n-1 = 24

# t scores of 96% confidence interval for sample size of 25
t.ppf(0.96,24)

# t scores of 99% confidence interval for sample size of 25
t.ppf(0.99,24)

"""
Q5
A Government company claims that an average light bulb lasts 
270 days. A researcher randomly selects 18 bulbs for testing.
The sampled bulbs last an average of 260 days, with a standard 
deviation of 90 days. If the CEO's claim were true, what is the
probability that 18 randomly selected bulbs would have an average 
life of no more than 260 days
"""

# find t-scores at x=260; t=(s_mean-P_mean)/(s_SD/sqrt(n))
tval=(260-270)/(90/18**0.5)
tval

# p_value=1-stats.t.cdf(abs(t_scores),df=n-1)... Using cdf function
p_value = 1-t.cdf(abs(tval),df=17)
p_value


"""
Q6

 The time required for servicing transmissions is normally 
 distributed with  = 45 minutes and  = 8 minutes. The service 
 manager plans to have work begin on the transmission of a customer’s 
 car 10 minutes after the car is dropped off and the
 customer is told that the car will be ready within 1 hour from drop-off.
 What is the probability that the service manager cannot meet his commitment?
 """
 
# Find Z-Scores at X=50; Z = (X - µ) / σ 
Zval=(50-45)/8
Zval

# Find probability P(X>50) 
1-norm.cdf(abs(Zval))

"""
Q7
A.	More employees at the processing center are older than 44 than 
between 38 and 44.
"""

# p(X>44); Employees older than 44 yrs of age
1-norm.cdf(44,38,6)

# p(38<X<44); Employees between 38 to 44 yrs of age
norm.cdf(44,38,6)-norm.cdf(38,38,6)

"""
A training program for employees under the age of 30 
at the center would be expected to attract about 36 employees.
"""
# P(X<30); Employees under 30 yrs of age
pvalue = norm.cdf(30,38,6)

# No. of employees attending training program from 400 nos. is N*P(X<30)
numemp = pvalue * 400
numemp


"""
Q8
If X1 ~ N(μ, σ2) and X2 ~ N(μ, σ2) are iid normal random variables
Then 2X1 ~ N(2μ, 4σ2)
And X1 + X2 ~ N(2μ, 2σ2)
"""


"""
Q9
Let X ~ N(100, 20^2) its (100, 20 square).Find two values, a and b, 
symmetric about the mean, such that the probability of the random 
variable taking a value between them is 0.99.
"""

norm.interval(0.99,100,20)

#Q10
# all answers in million 

# Mean profits from two different divisions of a company = Mean1 + Mean2
mean = (5+7)*45
mean

# Variance of profits from two different divisions of a company = SD^2 = SD1^2 + SD2^2
sd = (np.sqrt((9)+(16)))*45
sd

# A. Specify a Rupee range (centered on the mean) such that it contains 95% probability for the annual profit of the company.
norm.interval(0.95,mean,sd)

# B. Specify the 5th percentile of profit (in Rupees) for the company
# To compute 5th Percentile, we use the formula X=μ + Zσ; wherein from z table, 5 percentile = -1.645
X= mean+(-1.645)*(sd)
X

# C. Which of the two divisions has a larger probability of making a loss in a given year?
# Probability of Division 1 making a loss P(X<0)
norm.cdf(0,5,3)

# Probability of Division 2 making a loss P(X<0)
norm.cdf(0,7,4)
