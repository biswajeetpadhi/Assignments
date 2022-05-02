#STANDARDIZATION 

# Import required libraries
import pandas as pd
import numpy as np

# reading csv file
dataset_std = pd.read_csv("E:\\A3\\DataSets-Data Pre Processing\\DataSets\\Seeds_data.csv")
from sklearn.preprocessing import StandardScaler

#initilize scaler
scaler = StandardScaler() 

#to scale data
std=scaler.fit_transform(dataset_std)

#convert array to dataframe
dataset_new=pd.DataFrame(std)
dataset_new.describe()

#NORMALIZATION

# Import required libraries
import pandas as pd

# reading csv file
dataset_norm=pd.read_csv("E:\\A3\\DataSets-Data Pre Processing\\DataSets\\Seeds_data.csv")

#create a custom function
def norm_func(i):
                x= (i-i.min())/i.max()-i.min()
                return(x)
            
dataset_normalised=norm_func(dataset_norm)
dataset_normalised.describe()
