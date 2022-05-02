# TRANSFORMATIONS

# Import required libraries
import pandas as pd
import numpy as np

# reading csv file
dataset_trans = pd.read_csv("E:\\A3\\DataSets-Data Pre Processing\\DataSets\\calories_consumed.csv")

#natural logarithm
dataset_trans['log'] = np.log(dataset_trans['Calories Consumed'])

#logarithm of base 10
dataset_trans['log10'] = np.log10(dataset_trans['Weight gained (grams)'])

# finding exponential
dataset_trans['expo_value_weight'] = np.exp(dataset_trans['Weight gained (grams)'])
