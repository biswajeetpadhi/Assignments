# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 18:27:42 2021

@author: biswa
"""

import pandas as pd
import networkx as nx 

# Degree Centrality
G = pd.read_csv("E:\\ASSIGNMENT\\Network Analytics\\Datasets_Network Analytics\\routes.csv")

G = G.iloc[:, 1:]

g = nx.Graph()

g = nx.from_pandas_edgelist(G, source = 'Source Airport', target = 'Destination Airport')

print(nx.info(g))

b = nx.degree_centrality(g)  # Degree Centrality
print(b) 


# closeness centrality
closeness = nx.closeness_centrality(g)
print(closeness)

## Betweeness Centrality 
bt = nx.betweenness_centrality(g) # Betweeness_Centrality
print(bt)


