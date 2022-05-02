

# Hierarchical Clustering
#EastWestAirlines

import pandas as pd
import matplotlib.pylab as plt

airlines = pd.read_excel("E:\\ASSIGNMENT\\A6\\Dataset_Assignment Clustering\\EastWestAirlines.xlsx")

airlines.describe()
airlines.info()

airline = airlines.drop(["ID#"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
airline_norm = norm_func(airline.iloc[:,:])
airline_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
import matplotlib.pylab as plt


a = linkage(airline_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(a, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

hier_clus = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(airline_norm) 
hier_clus.labels_

cluster_labels = pd.Series(hier_clus.labels_)

airline['clust'] = cluster_labels # creating a new column and assigning it to new column 

airlines= airline.iloc[:, [11,0,1,2,3,4,5,6,7,8,9,10]]
airlines.head()

# Aggregate mean of each cluster
airlines.iloc[:, 1:].groupby(airlines.clust).mean()

# creating a csv file 
airlines.to_csv("Airlines.csv", encoding = "utf-8")

import os
os.getcwd()






# Hierarchical Clustering
#crime_data

import pandas as pd
import matplotlib.pylab as plt

crime = pd.read_csv("E:\\ASSIGNMENT\\A6\\Dataset_Assignment Clustering\\crime_data.csv")

crime.describe()
crime.info()

crime_new = crime.drop(["name"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
crime_norm = norm_func(crime_new.iloc[:,:])
crime_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
import matplotlib.pylab as plt


c = linkage(crime_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(c, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

crime_clus = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(airline_norm) 
crime_clus.labels_

cluster_labels = pd.Series(crime_clus.labels_)

crime_new['clust'] = cluster_labels # creating a new column and assigning it to new column 

crime= crime_new.iloc[:, [4,0,1,2,3]]
crime.head()

# Aggregate mean of each cluster
crime.iloc[:, 1:].groupby(crime.clust).mean()

# creating a csv file 
crime.to_csv("Crime.csv", encoding = "utf-8")

import os
os.getcwd()


# Hierarchical Clustering
#Telco_customer_churn

import pandas as pd
import matplotlib.pylab as plt

telco = pd.read_excel("E:\\ASSIGNMENT\\A6\\Dataset_Assignment Clustering\\Telco_customer_churn.xlsx")

telco.describe()
telco.info()

#one hot encoding
from sklearn.preprocessing import OneHotEncoder

enco= OneHotEncoder() # initializing method
telco_dum= pd.DataFrame(enco.fit_transform(telco.iloc[:,:]).toarray())


# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
import matplotlib.pylab as plt


t = linkage(telco_dum, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(t, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

telco_clus = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(telco_dum) 
telco_clus.labels_

cluster_labels = pd.Series(telco_clus.labels_)

telco['clust'] = cluster_labels # creating a new column and assigning it to new column 

telco1= telco.iloc[:, [30,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]]
telco1.head()

# Aggregate mean of each cluster
telco1.iloc[:, 1:].groupby(telco1.clust).mean()

# creating a csv file 
telco1.to_csv("Telco1.csv", encoding = "utf-8")

import os
os.getcwd()






# Hierarchical Clustering
#Autoinsurance

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

insurance = pd.read_csv("E:\\ASSIGNMENT\\A6\\Dataset_Assignment Clustering\\AutoInsurance.csv")

insurance.describe()
insurance.info()


#one hot encoding
from sklearn.preprocessing import OneHotEncoder

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

insurance['Customer']= label_encoder.fit_transform(insurance['Customer'])
insurance['State']= label_encoder.fit_transform(insurance['State'])
insurance['Response']= label_encoder.fit_transform(insurance['Response'])
insurance['Coverage']= label_encoder.fit_transform(insurance['Coverage'])
insurance['Education']= label_encoder.fit_transform(insurance['Education'])
insurance['Effective To Date']= label_encoder.fit_transform(insurance['Effective To Date'])
insurance['EmploymentStatus']= label_encoder.fit_transform(insurance['EmploymentStatus'])
insurance['Gender']= label_encoder.fit_transform(insurance['Gender'])
insurance['Location Code']= label_encoder.fit_transform(insurance['Location Code'])
insurance['Marital Status']= label_encoder.fit_transform(insurance['Marital Status'])
insurance['Policy Type']= label_encoder.fit_transform(insurance['Policy Type'])
insurance['Policy']= label_encoder.fit_transform(insurance['Policy'])
insurance['Renew Offer Type']= label_encoder.fit_transform(insurance['Renew Offer Type'])
insurance['Sales Channel']= label_encoder.fit_transform(insurance['Sales Channel'])
insurance['Vehicle Class']= label_encoder.fit_transform(insurance['Vehicle Class'])
insurance['Vehicle Size']= label_encoder.fit_transform(insurance['Vehicle Size'])


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
insurance_norm = norm_func(insurance.iloc[:,:])
insurance_norm.describe()


# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
import matplotlib.pylab as plt


i = linkage(insurance_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(i, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 5 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

insurance_clus = AgglomerativeClustering(n_clusters = 5, linkage = 'complete', affinity = "euclidean").fit(insurance_norm) 
insurance_clus.labels_

cluster_labels = pd.Series(insurance_clus.labels_)

insurance['clust'] = cluster_labels # creating a new column and assigning it to new column 

insurance= insurance.iloc[:, [24,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,23]]
insurance.head()

# Aggregate mean of each cluster
insurance.iloc[:, 1:].groupby(insurance.clust).mean()

# creating a csv file 
insurance.to_csv("Insurance.csv", encoding = "utf-8")

import os
os.getcwd()

