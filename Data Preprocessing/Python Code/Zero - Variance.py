#Zero - Variance 

# Import required libraries
import pandas as pd
import numpy as np

# reading csv file
dataset_var = pd.read_csv("E:\\A3\\DataSets-Data Pre Processing\\DataSets\\Z_dataset.csv")
dataset_var.var()

#row variance
dataset_var.var(axis=1)

#column variance
dataset_var.var(axis=0)
