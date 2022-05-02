#Dummy Variable

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reading csv file
dumvar = pd.read_csv("E:\\A3\\DataSets-Data Pre Processing\\DataSets\\animal_category.csv")

#create dummy variables
dumvar_new =pd.get_dummies(dumvar)
dumvar_newq=pd.get_dummies(dumvar,drop_first = True)

#one hot encoding

# Import required libraries
import pandas as pd

from sklearn.preprocessing import OneHotEncoder

# reading csv file
dumvar = pd.read_csv("E:\\A3\\DataSets-Data Pre Processing\\DataSets\\animal_category.csv")
enco= OneHotEncoder() # initializing method
enco_dumvar= pd.DataFrame(enco.fit_transform(dumvar.iloc[:,1:]).toarray())
