# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 12:02:33 2021

@author: biswa
"""

import numpy as np
import pandas as pd

tweet = pd.read_csv("E:\\ASSIGNMENT\\NaiveBayes\\Datasets_Naive Bayes\\Disaster_tweets_NB.csv")


x = tweet.iloc[:, 3]
y = tweet.iloc[:, 4].values


import nltk
nltk.download('stopwords')

# cleaning data 
import re
stop_words = []
# Load the custom built Stopwords
with open("E:\\ASSIGNMENT\\NaiveBayes\\stopwords.txt","r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")
   
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

x = x.apply(cleaning_text)

x = pd.DataFrame(x)

# removing empty rows
x = x.loc[x.text != " ",:]

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# splitting data into train and test data sets 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

y_train = pd.DataFrame(y_train)
y_train.columns =['target']
y_test = pd.DataFrame(y_test)
y_test.columns =['target']

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


# Defining the preparation of tweet texts into word count matrix format - Bag of Words
x_bow = CountVectorizer(analyzer = split_into_words).fit(x.text)

# Defining BOW for all messages
x_matrix = x_bow.transform(x.text)

# For training messages
train_matrix = x_bow.transform(x_train.text)

# For testing messages
test_matrix = x_bow.transform(x_test.text)

# Learning Term weighting and normalizing on entire tweet
tfidf_transformer = TfidfTransformer().fit(x_matrix)

# Preparing TFIDF for train tweet
train_tfidf = tfidf_transformer.transform(train_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test tweet
test_tfidf = tfidf_transformer.transform(test_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 
#Multinomial Naive Bayes


# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, y_train.target)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == y_test.target)
accuracy_test_m
