# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 15:33:22 2022

@author: biswa
"""

# Latent Dirichlet Allocation

import pandas as pd
# import warnings

tweets = pd.read_csv('E:\\ASSIGNMENT DS\\NLP - Topic Modeling\\NLP-TM/Data.csv', usecols=['text'])

import re

HANDLE = '@\w+'
LINK = 'https?://t\.co/\w+'
SPECIAL_CHARS = '&lt;|&lt;|&amp;|#'

def clean(text):
    text = re.sub(HANDLE, ' ', text)
    text = re.sub(LINK, ' ', text)
    text = re.sub(SPECIAL_CHARS, ' ', text)
    return text

tweets['text'] = tweets.text.apply(clean)

# LDA
from gensim.parsing.preprocessing import preprocess_string

tweets = tweets.text.apply(preprocess_string).tolist()

from gensim import corpora
from gensim.models.ldamodel import LdaModel

dictionary = corpora.Dictionary(tweets)
corpus = [dictionary.doc2bow(text) for text in tweets]

NUM_TOPICS = 5
ldamodel = LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=10)

ldamodel.print_topics(num_words=5)

from gensim.models.coherencemodel import CoherenceModel

def calculate_coherence_score(documents, dictionary, model):
    coherence_model = CoherenceModel(model=model, texts=documents, dictionary=dictionary, coherence='c_v')
    return coherence_model.get_coherence()

def get_coherence_values(start, stop):
    for num_topics in range(start, stop):
        print(f'\nCalculating coherence for {num_topics} topics')
        ldamodel = LdaModel(corpus, num_topics = num_topics, id2word=dictionary, passes=2)
        coherence = calculate_coherence_score(tweets, dictionary, ldamodel)
        yield coherence


min_topics, max_topics = 10,12
coherence_scores = list(get_coherence_values(min_topics, max_topics))

import matplotlib.pyplot as plt


x = [int(i) for i in range(min_topics, max_topics)]

ax = plt.figure(figsize=(10,8))
plt.xticks(x)
plt.plot(x, coherence_scores)
plt.xlabel('Number of topics')
plt.ylabel('Coherence Value')
plt.title('Coherence Scores', fontsize=10);

