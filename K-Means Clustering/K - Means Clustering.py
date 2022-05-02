#K - Means Clustering
#airlines




import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans

airlines= pd.read_excel("E:\\ASSIGNMENT\\A7\\Datasets_Kmeans\\EastWestAirlines (1).xlsx")

airlines.describe()

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
airlines_norm = norm_func(airlines.iloc[:, 1:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 7))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(airlines_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting  4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(airlines_norm)

model.labels_ # getting the labels of clusters assigned to each row 
ar = pd.Series(model.labels_)  # converting numpy array into pandas series object 
airlines['clust'] = ar # creating a  new column and assigning it to new column 

airlines.head()
airlines.info()


airlines = airlines.iloc[:,[12,0,1,2,3,4,5,6,7,8,9,10,11]]
airlines.head()

airlines.iloc[:, 2:].groupby(airlines.clust).mean()

airlines.to_csv("airlines.csv", encoding = "utf-8")

import os
os.getcwd()







#crime


import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans

crime= pd.read_csv("E:\\ASSIGNMENT\\A7\\Datasets_Kmeans\\crime_data (1).csv")

crime.describe()
crime.info()

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
crime_norm = norm_func(crime.iloc[:, 1:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 7))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(crime_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting  4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(crime_norm)

model.labels_ # getting the labels of clusters assigned to each row 
cr = pd.Series(model.labels_)  # converting numpy array into pandas series object 
crime['clust'] = cr # creating a  new column and assigning it to new column 

crime.head()
crime.info()


crime = crime.iloc[:,[5,0,1,2,3,4]]
crime.head()

crime.iloc[:, 2:].groupby(crime.clust).mean()

crime.to_csv("crime.csv", encoding = "utf-8")

import os
os.getcwd()






#Insurance


import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans

insurance= pd.read_csv("E:\\ASSIGNMENT\\A7\\Datasets_Kmeans\\Insurance Dataset.csv")

insurance.describe()
insurance.info()

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
insurance_norm = norm_func(insurance.iloc[:,:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 7))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(insurance_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting  4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(insurance_norm)

model.labels_ # getting the labels of clusters assigned to each row 
ins = pd.Series(model.labels_)  # converting numpy array into pandas series object 
insurance['clust'] = ins # creating a  new column and assigning it to new column 

insurance.head()
insurance.info()


insurance = insurance.iloc[:,[5,0,1,2,3,4]]
insurance.head()

insurance.iloc[:, 1:].groupby(insurance.clust).mean()

insurance.to_csv("insurance.csv", encoding = "utf-8")

import os
os.getcwd()







#telco cuatomers


import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans

telco= pd.read_excel("E:\\ASSIGNMENT\\A7\\Datasets_Kmeans\\Telco_customer_churn (1).xlsx")

telco.describe()
telco.info()

#one hot encoding
from sklearn.preprocessing import OneHotEncoder

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

telco['Customer ID']= label_encoder.fit_transform(telco['Customer ID'])
telco['Quarter']= label_encoder.fit_transform(telco['Quarter'])
telco['Offer']= label_encoder.fit_transform(telco['Offer'])
telco['Referred a Friend']= label_encoder.fit_transform(telco['Referred a Friend'])
telco['Phone Service']= label_encoder.fit_transform(telco['Phone Service'])
telco['Multiple Lines']= label_encoder.fit_transform(telco['Multiple Lines'])
telco['Internet Service']= label_encoder.fit_transform(telco['Internet Service'])
telco['Internet Type']= label_encoder.fit_transform(telco['Internet Type'])
telco['Online Security']= label_encoder.fit_transform(telco['Online Security'])
telco['Online Backup']= label_encoder.fit_transform(telco['Online Backup'])
telco['Device Protection Plan']= label_encoder.fit_transform(telco['Device Protection Plan'])
telco['Premium Tech Support']= label_encoder.fit_transform(telco['Premium Tech Support'])
telco['Streaming TV']= label_encoder.fit_transform(telco['Streaming TV'])
telco['Streaming Movies']= label_encoder.fit_transform(telco['Streaming Movies'])
telco['Streaming Music']= label_encoder.fit_transform(telco['Streaming Music'])
telco['Unlimited Data']= label_encoder.fit_transform(telco['Unlimited Data'])
telco['Contract']= label_encoder.fit_transform(telco['Contract'])
telco['Paperless Billing']= label_encoder.fit_transform(telco['Paperless Billing'])
telco['Payment Method']= label_encoder.fit_transform(telco['Payment Method'])



# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
telco_norm = norm_func(telco.iloc[:,3:])

np.any(np.isnan(telco_norm))
np.all(np.isfinite(telco_norm))

telco_norm.isna().sum()


###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 7))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(telco_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting  3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(telco_norm)

model.labels_ # getting the labels of clusters assigned to each row 
te = pd.Series(model.labels_)  # converting numpy array into pandas series object 
telco['clust'] = te # creating a  new column and assigning it to new column 

telco.head()
telco.info()


telco = telco.iloc[:,[30,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]]
telco.head()

telco.iloc[:, 4:].groupby(telco.clust).mean()

telco.to_csv("telco.csv", encoding = "utf-8")

import os
os.getcwd()







#autoinsurance


import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans

auto= pd.read_csv("E:\\ASSIGNMENT\\A7\\Datasets_Kmeans\\AutoInsurance (1).csv")

auto.describe()
auto.info()

#one hot encoding
from sklearn.preprocessing import OneHotEncoder

# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

auto['Customer']= label_encoder.fit_transform(auto['Customer'])
auto['State']= label_encoder.fit_transform(auto['State'])
auto['Response']= label_encoder.fit_transform(auto['Response'])
auto['Coverage']= label_encoder.fit_transform(auto['Coverage'])
auto['Education']= label_encoder.fit_transform(auto['Education'])
auto['Effective To Date']= label_encoder.fit_transform(auto['Effective To Date'])
auto['EmploymentStatus']= label_encoder.fit_transform(auto['EmploymentStatus'])
auto['Gender']= label_encoder.fit_transform(auto['Gender'])
auto['Location Code']= label_encoder.fit_transform(auto['Location Code'])
auto['Marital Status']= label_encoder.fit_transform(auto['Marital Status'])
auto['Policy Type']= label_encoder.fit_transform(auto['Policy Type'])
auto['Policy']= label_encoder.fit_transform(auto['Policy'])
auto['Renew Offer Type']= label_encoder.fit_transform(auto['Renew Offer Type'])
auto['Sales Channel']= label_encoder.fit_transform(auto['Sales Channel'])
auto['Vehicle Class']= label_encoder.fit_transform(auto['Vehicle Class'])
auto['Vehicle Size']= label_encoder.fit_transform(auto['Vehicle Size'])



# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
auto_norm = norm_func(auto.iloc[:,:])

np.any(np.isnan(auto_norm))
np.all(np.isfinite(auto_norm))

auto_norm.isna().sum()


###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 7))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(auto_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting  3 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(auto_norm)

model.labels_ # getting the labels of clusters assigned to each row 
au = pd.Series(model.labels_)  # converting numpy array into pandas series object 
auto['clust'] = au # creating a  new column and assigning it to new column 

auto.head()
auto.info()


auto = auto.iloc[:,[24,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]
auto.head()

auto.iloc[:, 4:].groupby(auto.clust).mean()

auto.to_csv("auto.csv", encoding = "utf-8")

import os
os.getcwd()
