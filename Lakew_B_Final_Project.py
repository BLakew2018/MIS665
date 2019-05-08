#!/usr/bin/env python
# coding: utf-8

# # Final Project - Has   Boeing 737 MaxA impacted other tourism sectors?

# ## Betelihem Lakew
# ### On my honor, as a student, I have neither given nor received unauthorized aid on this academic work.

# ### Introduction

# - After two deadly crashes in Singapore and Ethiopia, Boeing 737 Max A is grounded worldwide until the mid of summer. This has enormous financial impact which has already realized both in Boeing and the Airlines which uses this type of plane. However, how this would impact the whole tourism is yet to be determined.
# 

# ## Methodology

# - In order to understand the consumer confidence on traveling over 4000 real time tweets are collected using API and about 2000 historical data are collected using NodeXL.

# # Twitter Data Collection - Boeing

# # Retrieve and Process Twitter Data in "List"
# 
# - For this project we use various Python packages to retrieve tweets and cleaning it up and present it in a meaningful way
# 
# 

# In[51]:


# import popular packages
import csv
import pandas as pd
import re
import json
from scipy import stats

from collections import Counter

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist, word_tokenize

from os import path
from wordcloud import WordCloud, STOPWORDS

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from nltk.sentiment.vader import SentimentIntensityAnalyzer
vs = SentimentIntensityAnalyzer()
from textblob import TextBlob 

from operator import itemgetter

from IPython.display import Image


# In[52]:


import json

# create an empty list to store our tweets in
data = []

# append each line of the data to our tweets list using the json module
for line in open('Boeing 737_Max.json'):
    try:
        data.append(json.loads(line))
    except:
        pass

# lets see how many we got
print(len(data))


# - For the Project about 4642 tweets are collected in real time. In addition, another 2000 historical data is also collected which we will see later.

# In[53]:


tweetdata =[]
for i in data[:10]:
    print(i)
    tweetdata.append(i)
print(tweetdata)


# In[54]:


df = pd.DataFrame(tweetdata)
df.head(2)


# ## Data Cleaning

# - The cleaning process starts my removing all message( error message, for instance) to make sure the data collected are entirely tweets.

# In[4]:


# The filtered data is then saved in a variable and print it

texts = [ T['text'] for T in data if 'text' in T ]
len(texts)


# In[5]:


df = pd.DataFrame(texts)
df.head(2)


# In[6]:


for T in data:
    if 'text' not in T:
        print(T)


# In[7]:


for i in data[:5]:
    print(i)


# In[8]:



tweets = []
for T in data:
    if 'text' in T:
        tweets.append(T)
len(tweets)   


# #### Once we make sure the retrieved data is entirely tweets, we will start looking for meaningful information.

# In[9]:


# display screen names (twitter user names)
screen_names = [T['user']['screen_name'] for T in tweets]

(screen_names)


# In[10]:


# save screen_names

screen_names = [T['user']['screen_name'] for T in tweets]
len(screen_names)


# In[12]:


# display screen_name, tweets
tweettext = []
for i in tweets[:10]:
    print(i['user']['screen_name'], i['text'])
    tweettext.append(i)
print(tweettext)


# # We Can Process Tweets in Dataframe: Parse JSON to CSV Automatically

# In[13]:


# Extracting information from tweets using Python codes

ids = [T['id_str'] for T in tweets]
times = [T['created_at'] for T in tweets]
texts = [T['text'] for T in tweets]
screen_names = [T['user']['screen_name'] for T in tweets]
followers_count = [T['user']['followers_count'] for T in tweets]
friends_count = [T['user']['friends_count'] for T in tweets]
names = [T['user']['name'] for T in tweets]
lats = [(T['geo']['coordinates'][0] if T['geo'] else None) for T in tweets]
lons = [(T['geo']['coordinates'][1] if T['geo'] else None) for T in tweets]
place_names = [(T['place']['full_name'] if T['place'] else None) for T in tweets]
place_types = [(T['place']['place_type'] if T['place'] else None) for T in tweets]

# open an output csv file to write to
out = open('boeing737data.csv', 'w', encoding='UTF-8', newline='')

# write the header of our CSV as its first line
out.write('id,created at,text,screen name,followers_count,friends_count,name,lat,lon,place name,place type\n')

# merge each individual list into a single list using the zip function
rows = list(zip(ids, times, texts, screen_names, followers_count, friends_count, names, lats, lons, place_names, place_types))

# use the writer module on our csv file
csv = csv.writer(out)

# use one value from each of our rows list and write it to the csv as a new row
for row in rows:
    values = [value for value in row]
    #values = [(value.encode('utf8') if hasattr(value, 'encode') else value) for value in row]
    csv.writerow(values)

# close our csv file when done
out.close()

#http://mike.teczno.com/notes/streaming-data-from-twitter.html


# In[14]:


# The extracted data as dataframe

df = pd.read_csv("boeing737data.csv")
df.head()


# In[18]:


influencers = df.sort_values('followers_count', ascending=False).head(10)
influencers.head(5)


# In[ ]:


- Here 


# # Descriptive Analytics
# 
# We will process and analyze the data in **list**
# 
# Of course, you can do these activities in **dataframe** as well (**Module "Pansas for Text Analytics"**)

# ### Tweets per user

# In[19]:


from collections import Counter

c = Counter(screen_names)
print(c)


# In[20]:


# how many unique users in the data?
len(c)


# In[21]:


#how many tweets per user?

float(20/20)


# ### Most active users

# In[22]:


from collections import Counter

c = Counter(screen_names)
print(c)


# In[23]:


# five most active tweeters
# make it pretty
activetweeters = c.most_common(5)
activetweeters_df = pd.DataFrame(activetweeters)
activetweeters_df


# ### Popular languages

# In[24]:


lang = [T['user']['lang'] for T in tweets if 'user' in T]

c = Counter(lang)
print(c)


# #### The above data shows even though there are 27 languages are represented in the most significant language represented is english. This guide the project in two ways
# - 1. Using only english data for further research is able to give an accurate data
# - 2. As tweeter is viewed globally local newspapers can impact not only locals but the international audience influence global consumers of American goods and services

# In[55]:


# extract all english tweets & meta data and save

english =[]
for i in tweets:
    if i['user']['lang'] == 'en':
        english.append(i)
len(english)


# In[57]:


# read first2 English tweets only 
englishtweets = []
for i in english[:2]:
    print(i['text'])
    englishtweets.append(i['text'])
print(englishtweets)


# ### Most visible users 

# In[80]:


for tweet in texts[:10]:
    print(tweet)


# In[81]:


# first extract all users from tweets

#let's use regular expression ... 
    
import re

for tweet in texts[:5]:
    print(re.findall(r"(?<=@)\w+", tweet))


# In[82]:


for tweet in texts[:5]:
    a = re.findall(r"(?<=@)\w+", tweet)
    for i in a:
        print('@'+i)


# In[83]:


visible_users = []

for tweet in texts:
    a = re.findall(r"(?<=@)\w+", tweet)
    for i in a:
        visible_users.append(['@'+i][0])


# In[84]:


for i in visible_users[:5]:
    print(i)


# In[85]:


# 10 most visible users in this dataset

c = Counter(visible_users)
c.most_common(10)


# ### Findings of Real time data

# - So far there is no indicator that consumers are discussing family vacations as a dependent factor on recent incidents of Boeing. New Findings however is the american media like WSJ and CNN spen as well as other news outlets are dominant network as they have a leading number of followers.  

# ### Analyzing historical data
# - This Project has accumulated historical data ( Tweets data which are not collected in real time but which are couple of days older) and analyzed it using NodeXL

# In[94]:


histdata = pd.read_csv('data/TweetStat.csv')
histdata.head(5)


# - In this analytics it is shown that negative sentiment is dominating which impact Boeing but not necessarily tourism or travel

# In[96]:


histdata2 = pd.read_csv('data/Toptweet.csv')
histdata2.head(5)


#  - The Top word Pairs indicate which words are mentioned frequently. The result does not show any word correlation between Boeing and Vacation plans

# ### Conclusion
# - The recent crashes of Boeing 737maxA has surely cost the commony a severe financial damage and this social media offers a glimps of the damage caused on the image and reputation of the company and its management. In addition, this Sentiment analysis reveals that news outlets are influencing networks in the social media and can be a reliable source of information.

# ## Betelihem Lakew
# ### On my honor, as a student, I have neither given nor received unauthorized aid on this academic work.
