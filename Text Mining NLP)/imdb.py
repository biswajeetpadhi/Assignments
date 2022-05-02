# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 16:25:57 2022

@author: biswa
"""


import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re # regular expressions 
import matplotlib.pyplot as plt
from wordcloud import WordCloud

shawshank =[]
for i in range(1,21):
  imdb=[]  
  url = "https://www.imdb.com/title/tt0111161/reviews?ref_=tt_sa_3"+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.findAll("div",attrs={"class","text show-more__control"})
  # Extracting the content under specific tags  
  
  for i in range(len(reviews)):
    imdb.append(reviews[i].text)  
  shawshank=shawshank+imdb 
  
  
  
# Joinining all the reviews into single paragraph 
shawshank_string = " ".join(shawshank)



# Removing unwanted symbols incase if exists
shawshank_string = re.sub("[^A-Za-z" "]+"," ",shawshank_string).lower()
shawshank_string = re.sub("[0-9" "]+"," ",shawshank_string)

# words that contained in The Shawshank Redemption reviews
shawshank_words = shawshank_string.split(" ")

#stop_words = stopwords.words('english')

with open("E:\\ASSIGNMENT DS\\Text Mining NLP)\\Datasets NLP\\stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")

shawshank_words = [w for w in shawshank_words if w not in stopwords]

# Joinining all the reviews into single paragraph 
shawshank_string = " ".join(shawshank_words)
#forming wordcloud
wordcloud_TSR = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(shawshank_string)

plt.imshow(wordcloud_TSR)

#positive word reviews
with open("E:\\ASSIGNMENT DS\\Text Mining NLP)\\Datasets NLP\\positive-words.txt","r") as positive:
  positivewords = positive.read().split("\n")
  
positivew = positivewords[35:]

shawshankpositive = " ".join ([w for w in shawshank_words if w in positivew])

shawshankpositivecloud = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(shawshankpositive)

plt.imshow(shawshankpositivecloud)


# negative word reviews
with open("E:\\ASSIGNMENT DS\\Text Mining NLP)\\Datasets NLP\\negative-words.txt","r") as negative:
  negativewords = negative.read().split("\n")

negativewords = negativewords[35:]

shawshanknegative = " ".join ([w for w in shawshank_words if w in negativewords])

shawshanknegativecloud = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(shawshanknegative)

plt.imshow(shawshanknegativecloud)
