# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 19:04:52 2022

@author: biswa
"""

from scipy.stats import norm

#Q3
#Calculate 94%,98%,96% confidence interval?

# Avg. weight of Adult in Mexico with 94% CI
norm.interval(0.94,200,30/(2000**0.5))

# Avg. weight of Adult in Mexico with 98% CI
norm.interval(0.98,200,30/(2000**0.5))

# Avg. weight of Adult in Mexico with 96% CI
norm.interval(0.96,200,30/(2000**0.5))

