# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 16:35:00 2021

@author: biswa
"""

#Dimension Reduction With PCA



# k means
#heart

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans

heart_k = pd.read_csv("E:\\ASSIGNMENT\\PCA\\Datasets_PCA\\heart disease.csv")

heart_k.describe()
heart_k.info()


# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
heart_k_norm = norm_func(heart_k.iloc[:,:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 7))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(heart_k_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');
plt.xlabel("No_of_Clusters");
plt.ylabel("total_within_SS")

# Selecting  4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(heart_k_norm)

model.labels_ # getting the labels of clusters assigned to each row 
ht = pd.Series(model.labels_)  # converting numpy array into pandas series object 
heart_k['clust'] = ht # creating a  new column and assigning it to new column 

heart_k.head()
heart_k.info()


heart_k = heart_k.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
heart_k.head()

heart_k.iloc[:, 2:].groupby(heart_k.clust).mean()

heart_k.to_csv("heart_kmeans_before.csv", encoding = "utf-8")

import os
os.getcwd()






#PCA


import pandas as pd
import numpy as np

heart_pca = pd.read_csv("E:\\ASSIGNMENT\\PCA\\Datasets_PCA\\heart disease.csv")

heart_pca.describe()

heart_pca.info()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 


# Normalizing the numerical data 
heart_pca_normal = scale(heart_pca)
heart_pca_normal

pca = PCA(n_components = 14)
pca_values = pca.fit_transform(heart_pca_normal)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

# PCA weights
pca.components_


# Cumulative variance 
var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5","comp6", "comp7", "comp8", "comp9", "comp10", "comp11","comp12", "comp13"

pca_data.to_csv("pca_data_heart.csv", encoding = "utf-8")

import os
os.getcwd()







# k means after pca
#heart

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans

heart_k = pd.read_csv("E:\\ASSIGNMENT\\PCA\\Datasets_PCA\\heart disease.csv")
heart_k_after = pd.read_csv("E:\\ASSIGNMENT\\PCA\\datasheet extracted\\heart\\pca_data_heart.csv")

heart_k_after.describe()
heart_k_after.info()


# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
heart_k_after_norm = norm_func(heart_k_after.iloc[:,:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 7))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(heart_k_after_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');
plt.xlabel("No_of_Clusters");
plt.ylabel("total_within_SS")

# Selecting  4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(heart_k_after_norm)

model.labels_ # getting the labels of clusters assigned to each row 
ht = pd.Series(model.labels_)  # converting numpy array into pandas series object 
heart_k['clust'] = ht # creating a  new column and assigning it to new column 

heart_k.head()
heart_k.info()


heart_k = heart_k.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
heart_k.head()

heart_k.iloc[:, 2:].groupby(heart_k.clust).mean()

heart_k.to_csv("heart_kmeans_after.csv", encoding = "utf-8")

import os
os.getcwd()



