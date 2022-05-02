#Exploratory Data Analysis

#import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#read the csv file
vehicle= pd.read_csv("E:\\A4\\Statistical Datasets\\Statistical Datasets\\Q1_a.csv")


vehicle.speed.skew() #skewed
vehicle.speed.kurt() #kurtosis
vehicle.speed.mean() #mean
vehicle.speed.median() #median
vehicle.speed.mode() #mode
vehicle.speed.var() #variance
vehicle.speed.std() #standard deviation
range1= max(vehicle.speed)-min(vehicle.speed)
range1 #range

#boxtplot for dection of outliers
sns.boxplot(vehicle.speed)

#histogram for skewness checking
plt.hist(vehicle.speed) #negatively skewed

vehicle.dist.kurt() #kurtosis
vehicle.dist.skew() #skewed 
vehicle.dist.mean() #mean
vehicle.dist.median() #median
vehicle.dist.mode() #mode
vehicle.dist.var() #variance
vehicle.dist.std() #standard deviation
range2= max(vehicle.dist)-min(vehicle.dist)
range2 #range

#boxtplot for dection of outliers
sns.boxplot(vehicle.dist)

#histogram for skewness checking
plt.hist(vehicle.dist) #positively skewed


#import required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#read the csv file
vehi= pd.read_csv("E:\\A4\\Statistical Datasets\\Statistical Datasets\\Q2_b.csv")

vehi.SP.skew() #skewed
vehi.SP.kurt() #kurtosis
vehi.SP.mean() #mean
vehi.SP.median() #median
vehi.SP.mode() #mode
vehi.SP.var() #variance
vehi.SP.std() #standard deviation
range3 = max(vehi.SP)-min(vehi.SP)
range3 #range

#boxtplot for dection of outliers
sns.boxplot(vehi.SP)

vehi.WT.skew() # skewed
vehi.WT.kurt() # kurtosis
vehi.WT.mean() #mean
vehi.WT.median() #median
vehi.WT.mode() #mode
vehi.WT.var() #variance
vehi.SP.std() #standard deviation
range4 = max(vehi.WT)-min(vehi.WT)
range4 #range

#boxtplot for dection of outliers
sns.boxplot(vehi.WT)


#Q3) Below are the scores obtained by a student in tests 
#34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56

#import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#create a data frame
data1 = [34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]
df = pd.DataFrame(data1, columns = ['score'])

df.score.mean() #mean
df.score.median() #median
df.score.mode() #mode
df.score.var() #variance
df.score.std() #standard deviation

#histogram for skewness checking
plt.hist(df.score) #positively skewed

#finding outliers
sns.boxplot(df.score) 

