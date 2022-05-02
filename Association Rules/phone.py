# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:01:35 2021

@author: biswa
"""


import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori

phones = pd.read_csv("E:\\ASSIGNMENT\\Association Rules\\Datasets_Association Rules\\myphonedata.csv")
phone= phones.iloc[:,3:]

frequent_itemsets = apriori(phone, min_support = 0.0075, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()