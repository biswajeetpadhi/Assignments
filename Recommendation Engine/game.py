# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 14:26:26 2021

@author: biswa
"""


# import os
import pandas as pd

# import Dataset 
game= pd.read_csv("E:\\ASSIGNMENT\\Recommendation Engine\\Datasets_Recommendation Engine\\game.csv", encoding = 'utf8')

game.shape # shape
game.columns


from sklearn.feature_extraction.text import TfidfVectorizer 
#term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect 
# - how important a word is to document in a collecion or corpus

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 


game["titles"].isnull().sum() 

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(game.titles)  
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

# creating a mapping of game name to index number 
game_index = pd.Series(game.index, index = game['titles']).drop_duplicates()

game_id = game_index["Diablo"]
game_id

def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the game index using its title 
    game_id = game_index[Name]
    
    # Getting the pair wise similarity score for all the game's with that 
   
    cosine_scores = list(enumerate(cosine_sim_matrix[game_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar game
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the game index 
    game_idx  =  [i[0] for i in cosine_scores_N]
    game_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar game and scores
    game_similar_show = pd.DataFrame(columns=["titles", "Score"])
    game_similar_show["titles"] = game.loc[game_idx, "titles"]
    game_similar_show["Score"] = game_scores
    game_similar_show.reset_index(inplace = True)  
  
    print (game_similar_show)
  

    
# Enter your anime and number of anime's to be recommended 
get_recommendations("Diablo", topN = 10)
game_index["Diablo"]
