# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 16:05:04 2021

@author: biswa
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.frequent_patterns import apriori


store = []
with open("E:\\ASSIGNMENT\\Association Rules\\Datasets_Association Rules\\transactions_retail1.csv") as f:
   store = f.read()
    

# splitting the data into separate transactions using separator as "\n"
store = store.split("\n")

store_list = []
for i in store:
    store_list.append(i.split(","))


all_store_list = [i for item in store_list for i in item]


from collections import Counter # ,OrderedDict

item_frequencies = Counter(all_store_list)

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))


# Creating Data Frame for the transactions data
store_series = pd.DataFrame(pd.Series(store_list))
store_series = store_series.iloc[:501, :] # removing the last empty transaction

store_series.columns = ["transactions"]


# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = store_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')



frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

