#Inferential Statistics


# Q5) Calculate Mean, Median, Mode, Variance, Standard Deviation, Range & comment about the values / draw inferences, for the given dataset
#-	For Points, Score, Weigh>
#Find Mean, Median, Mode, Variance, Standard Deviation, and Range and comment about the values/ Draw some inferences. 

# Import required libraries
import pandas as pd


# reading csv file
data_stats= pd.read_excel("E:\\A3\\DataSets-Data Pre Processing\\DataSets\\Assignment_module.xlsx")

data_stats.Points.mean()
data_stats.Points.median()
data_stats.Points.mode()
data_stats.Points.var()
data_stats.Points.std()
range1= max(data_stats.Points)-min(data_stats.Points)
range1

data_stats.Score.mean()
data_stats.Score.median()
data_stats.Score.mode()
data_stats.Score.var()
data_stats.Score.std()
range2= max(data_stats.Score)-min(data_stats.Score)
range2

data_stats.Weigh.mean()
data_stats.Weigh.median()
data_stats.Weigh.mode()
data_stats.Weigh.var()
data_stats.Weigh.std()
range3= max(data_stats.Weigh)-min(data_stats.Weigh)
range3

#Q7) Look at the data given below. Plot the data, find the outliers and find out  μ,σ,σ^2
# Use a plot which shows the data distribution, skewness along with the outliers
# Import required libraries
import pandas as pd
import seaborn as sns

# reading csv file
data_infer= pd.read_excel("E:\\A3\\DataSets-Data Pre Processing\\DataSets\\Infer_stats.xlsx")
data_infer.Measure_X.mean()
data_infer.Measure_X.var()
data_infer.Measure_X.std()

#boxplot
sns.boxplot(data_infer.Measure_X)

#skewness
data_infer.Measure_X.skew()
