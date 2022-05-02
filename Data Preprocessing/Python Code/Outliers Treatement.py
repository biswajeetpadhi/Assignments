# Outlier Treatment

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# reading csv file
out1 = pd.read_csv("E:\\A3\\DataSets-Data Pre Processing\\DataSets\\boston_data.csv")

sns.boxplot(out1.age);plt.show()

#Outlier dection
IQR = out1['age'].quantile(0.75)-out1["age"].quantile(0.25)
low_limit = out1["age"].quantile(0.25)-(IQR*1.5)
up_limit = out1["age"].quantile(0.75)+(IQR*1.5)

#Trimming dataset
out_find = np.where(out1["age"]>up_limit,True ,
           np.where(out1["age"]<low_limit,True , False))
out_trimm= out1.loc[~(out_find),]

sns.boxplot(out_trimm.age);plt.show()

# replace outliers with max and min vlaue
out1["out1_replaced"]=pd.DataFrame(np.where(out1["age"]>up_limit,up_limit,
                                  np.where(out1["age"]<low_limit,low_limit,out1["age"])))

sns.boxplot(out1.out1_replaced)

#winsorization

# Import required libraries
import pandas as pd
import seaborn as sns

# reading csv file
out1 = pd.read_csv("E:\\A3\\DataSets-Data Pre Processing\\DataSets\\boston_data.csv")

from feature_engine.outliers import Winsorizer
winsor= Winsorizer(capping_method = "iqr",
                   tail = "both",
                   fold = 1.5,
                   variables= ["age"])
out_winz= winsor.fit_transform(out1[["age"]])

sns.boxplot(out_winz.age);plt.show()
