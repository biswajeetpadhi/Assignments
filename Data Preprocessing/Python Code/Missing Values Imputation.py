# Missing Values Imputation

# Import required libraries
import pandas as pd
import numpy as np

# reading csv file
missvalu = pd.read_csv("E:\\A3\\DataSets-Data Pre Processing\\DataSets\\claimants.csv")

#check missing values
missvalu.isna().sum()

from sklearn.impute import SimpleImputer

#mean  imputation
mean_impute=SimpleImputer(missing_values = np.nan,strategy="mean")
missvalu["CLMAGE"]=pd.DataFrame(mean_impute.fit_transform(missvalu[["CLMAGE"]]))
missvalu.isna().sum()

#median  imputation
median_impute=SimpleImputer(missing_values = np.nan,strategy="median")
missvalu["CLMINSUR"]=pd.DataFrame(median_impute.fit_transform(missvalu[["CLMINSUR"]]))
missvalu.isna().sum()

#mode  imputation
mode_impute=SimpleImputer(missing_values = np.nan,strategy="most_frequent")
missvalu["SEATBELT"]=pd.DataFrame(mode_impute.fit_transform(missvalu[["SEATBELT"]]))
missvalu.isna().sum() 