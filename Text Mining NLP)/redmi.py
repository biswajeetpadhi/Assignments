# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 17:20:50 2021

@author: biswa
"""

import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re # regular expressions 
import matplotlib.pyplot as plt
from wordcloud import WordCloud

redminote7all =[]

for i in range(1,21):
  redminote7=[]  
  url = "https://www.flipkart.com/redmi-note-7-pro-space-black-64-gb/product-reviews/itmfegkx2gufuzhp?pid=MOBFDXZ36Y4DJBGM&lid=LSTMOBFDXZ36Y4DJBGM2SHASI&marketplace=FLIPKART&page="+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.findAll("div",attrs={"class",""})
  # Extracting the content under specific tags  
  
  for i in range(len(reviews)):
    redminote7.append(reviews[i].text)  
  redminote7all=redminote7all+redminote7

# Joinining all the reviews into single paragraph 
redminote7all_string = " ".join(redminote7all)



# Removing unwanted symbols incase if exists
redminote7all_string = re.sub("[^A-Za-z" "]+"," ",redminote7all_string).lower()
redminote7all_string = re.sub("[0-9" "]+"," ",redminote7all_string)

# words that contained in redmi note 7 reviews
redminote7all_words = redminote7all_string.split(" ")


with open("E:\\ASSIGNMENT DS\\Text Mining NLP)\\Datasets NLP\\stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")

redminote7all_words = [w for w in redminote7all_words if w not in stopwords]

# Joinining all the reviews into single paragraph 
redminote7all_string = " ".join(redminote7all_words)
#forming wordcloud
wordcloud_n7 = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(redminote7all_string)

plt.imshow(wordcloud_n7)
#positive word reviews
with open("E:\\ASSIGNMENT DS\\Text Mining NLP)\\Datasets NLP\\positive-words.txt","r") as positive:
  positivewords = positive.read().split("\n")
  
positivew = positivewords[35:]

redminote7positive = " ".join ([w for w in redminote7all_words if w in positivew])

redminote7positivecloud = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(redminote7positive)

plt.imshow(redminote7positivecloud)


# negative word reviews
with open("E:\\ASSIGNMENT DS\\Text Mining NLP)\\Datasets NLP\\negative-words.txt","r") as negative:
  negativewords = negative.read().split("\n")

negativewords = negativewords[35:]

redminote7negative = " ".join ([w for w in redminote7all_words if w in negativewords])

redminote7negativecloud = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(redminote7negative)

plt.imshow(redminote7negativecloud)

