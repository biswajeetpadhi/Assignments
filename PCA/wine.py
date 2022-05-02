#Dimension Reduction With PCA

# Hierarchical Clustering
#wine

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

wine = pd.read_csv("E:\\ASSIGNMENT\\PCA\\Datasets_PCA\\wine.csv")

wine.describe()
wine.info()

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data))
wine_norm = norm_func(wine.iloc[:,:])
wine_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
import matplotlib.pylab as plt


i = linkage(wine_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));
plt.title('Hierarchical Clustering Dendrogram');
plt.xlabel('Index');
plt.ylabel('Distance')
sch.dendrogram(i, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 4   as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

wine_clus = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(wine_norm) 
wine_clus.labels_

cluster_labels = pd.Series(wine_clus.labels_)

wine['clust'] = cluster_labels # creating a new column and assigning it to new column 

wine.info()
wine= wine.iloc[:, [14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
wine.head()

# Aggregate mean of each cluster
wine.iloc[:, 1:].groupby(wine.clust).mean()

# creating a csv file 
wine.to_csv("wine_hier_before.csv", encoding = "utf-8")

import os
os.getcwd()






# k means
#wine

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans

wine_k = pd.read_csv("E:\\ASSIGNMENT\\PCA\\Datasets_PCA\\wine.csv")

wine_k.describe()
wine_k.info()


# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
wine_k_norm = norm_func(wine_k.iloc[:,:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 7))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(wine_k_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');
plt.xlabel("No_of_Clusters");
plt.ylabel("total_within_SS")

# Selecting  3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(wine_k_norm)

model.labels_ # getting the labels of clusters assigned to each row 
wi = pd.Series(model.labels_)  # converting numpy array into pandas series object 
wine_k['clust'] = wi # creating a  new column and assigning it to new column 

wine_k.head()
wine_k.info()


wine_k = wine_k.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
wine_k.head()

wine_k.iloc[:, 2:].groupby(wine_k.clust).mean()

wine_k.to_csv("wine_kmeans_before.csv", encoding = "utf-8")

import os
os.getcwd()






#PCA


import pandas as pd
import numpy as np

wine_pca = pd.read_csv("E:\\ASSIGNMENT\\PCA\\Datasets_PCA\\wine.csv")

wine_pca.describe()

wine_pca.info()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 


# Normalizing the numerical data 
wine_pca_normal = scale(wine_pca)
wine_pca_normal

pca = PCA(n_components = 14)
pca_values = pca.fit_transform(wine_pca_normal)

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

pca_data.to_csv("pca_data_wine.csv", encoding = "utf-8")

import os
os.getcwd()


# Hierarchical Clustering after pca
#wine

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

wine = pd.read_csv("E:\\ASSIGNMENT\\PCA\\Datasets_PCA\\wine.csv")
wine_after = pd.read_csv("E:\\ASSIGNMENT\\PCA\\datasheet extracted\\wine\\pca_data_wine.csv")

wine_after.describe()
wine_after.info()

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data))
wine_after_norm = norm_func(wine_after.iloc[:,:])
wine_after_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
import matplotlib.pylab as plt


i = linkage(wine_after_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(i, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

wine_clus = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(wine_after_norm) 
wine_clus.labels_

cluster_labels = pd.Series(wine_clus.labels_)

wine['clust'] = cluster_labels # creating a new column and assigning it to new column 

wine.info()
wine= wine.iloc[:, [14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
wine.head()

# Aggregate mean of each cluster
wine.iloc[:, 1:].groupby(wine.clust).mean()

# creating a csv file 
wine.to_csv("wine_hier_after.csv", encoding = "utf-8")

import os
os.getcwd()






# k means after pca
#wine

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans

wine_k = pd.read_csv("E:\\ASSIGNMENT\\PCA\\Datasets_PCA\\wine.csv")
wine_k_after = pd.read_csv("E:\\ASSIGNMENT\\PCA\\datasheet extracted\\wine\\pca_data_wine.csv")

wine_k_after.describe()
wine_k_after.info()


# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
wine_k_after_norm = norm_func(wine_k_after.iloc[:,:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 7))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(wine_k_after_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting  3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(wine_k_after_norm)

model.labels_ # getting the labels of clusters assigned to each row 
wi = pd.Series(model.labels_)  # converting numpy array into pandas series object 
wine_k['clust'] = wi # creating a  new column and assigning it to new column 

wine_k.head()
wine_k.info()


wine_k = wine_k.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
wine_k.head()

wine_k.iloc[:, 2:].groupby(wine_k.clust).mean()

wine_k.to_csv("wine_kmeans_after.csv", encoding = "utf-8")

import os
os.getcwd()



