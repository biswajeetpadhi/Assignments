#Duplication Typecasting

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# reading csv file
typecast = pd.read_csv("E:\\A3\\DataSets-Data Pre Processing\\DataSets\\OnlineRetail.csv")

#typecasting from float to int
typecast.UnitPrice=typecast.UnitPrice.astype("int64")

#identifying duplicates
duplicate=typecast.duplicated()
sum(duplicate)

#removing duplicate
typecast1=typecast.drop_duplicates()

#historgram
plt.hist(typecast.UnitPrice)

#boxplot
plt.boxplot(typecast.UnitPrice)
sns.boxplot(typecast.UnitPrice)
