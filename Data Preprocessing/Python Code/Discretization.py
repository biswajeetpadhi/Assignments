# Discreitization

# Import required libraries
import pandas as pd
import numpy as np

# reading csv file
rou = pd.read_csv("E:\\A3\\DataSets-Data Pre Processing\\DataSets\\iris.csv")

#rounding the values
rou['Sepal.Length'] = np.round(rou['Sepal.Length'], decimals = 0)
rou["Sepal.Width"] = np.round(rou["Sepal.Width"], decimals = 0)
rou['Petal.Length'] = np.round(rou['Petal.Length'], decimals = 0)
rou["Petal.Width"] = np.round(rou["Petal.Width"], decimals = 0)
rou

