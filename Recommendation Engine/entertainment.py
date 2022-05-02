# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 11:28:29 2021

@author: biswa
"""

# import os
import pandas as pd

# import Dataset 
entertainment= pd.read_csv("E:\\ASSIGNMENT\\Recommendation Engine\\Datasets_Recommendation Engine\\Entertainment.csv", encoding = 'utf8')

entertainment.shape # shape
entertainment.columns
entertainment.category 

from sklearn.feature_extraction.text import TfidfVectorizer 
#term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect 
# - how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 


entertainment["category"].isnull().sum() 

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(entertainment.category)  
 #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape 

# with the above matrix we need to find the similarity score
# There are several metrics for this such as the euclidean, 
# the Pearson and the cosine similarity scores

# For now we will be using cosine similarity matrix
# A numeric quantity to represent the similarity between 2 movies 
# Cosine similarity - metric is independent of magnitude and easy to calculate 

# cosine(x,y)= (x.y⊺)/(||x||.||y||)

from sklearn.metrics.pairwise import linear_kernel

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of anime name to index number 
entertainment_index = pd.Series(entertainment.index, index = entertainment['titles']).drop_duplicates()

entertainment_id = entertainment_index["Assassins (1995)"]
entertainment_id

def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the movie index using its title 
    entertainment_id = entertainment_index[Name]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[entertainment_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    entertainment_idx  =  [i[0] for i in cosine_scores_N]
    entertainment_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    entertainment_similar_show = pd.DataFrame(columns=["titles", "Score"])
    entertainment_similar_show["titles"] = entertainment.loc[entertainment_idx, "titles"]
    entertainment_similar_show["Score"] = entertainment_scores
    entertainment_similar_show.reset_index(inplace = True)  
  
    print (entertainment_similar_show)
   

    
# Enter your anime and number of anime's to be recommended 
get_recommendations("Toy Story (1995)", topN = 10)
entertainment_index["Toy Story (1995)"]
