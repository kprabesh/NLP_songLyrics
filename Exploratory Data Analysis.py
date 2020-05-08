#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# # 1. Top words in the lyrics
1. Top words in the lyrics of the five songs
2. Unique words used in the five songs
3. Level of profanity in the five snogs

In terminal type the following commands separately to install these packages: 
conda install -c conda-forge wordcloud 
conda install -c conda-forge textblob 
conda install -c conda-forge gensim 
# In[1]:


#Read document term matrix using pandas
import pandas as pd
data= pd.read_pickle("dtm.pkl")
data=data.transpose()
data.head(5)


# In[2]:


# Top 30 words on the basis of count in the lyrics of the songs
top_dict = {}
for c in data.columns:
    top = data[c].sort_values(ascending=False).head(30)
    top_dict[c]= list(zip(top.index, top.values))

top_dict


# In[3]:


# Take a look at Top 30 words again on the basis of word count in the song lyrics
for song, top_words in top_dict.items():
    print(song)
    print(', '.join([word for word, count in top_words[0:29]]))
    print('---')

Some of the words non sensical and can be removed to make the data moore meaningful and precise. 
So, we will add those words to stop word list before creating word clouds.
# In[4]:


#Look at the most common words among the five songs
from collections import Counter
''' pulling out top 30 words from the all lyrics'''

words = []
for song in data.columns:
    top = [word for (word, count) in top_dict[song]]
    for t in top:
        words.append(t)
        
words


# In[5]:


# Now we aggregate this list, and identify the most common words along with how many routines they occur in
Counter(words).most_common()


# In[6]:


# If more than half of the songs have it as a top word, exclude it from the list
add_stop_words = [word for word, count in Counter(words).most_common() if count > 3]
add_stop_words


# In[7]:


# Now we will update our document-term matrix with the new list of stop words
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer

# Reading cleaned data
data_clean = pd.read_pickle('data_clean.pkl')

# Adding new stop words from our top 30 words list to previous set 
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Creating document-term matrix again
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.Lyrics)
data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_stop.index = data_clean.index

# Pickle it for later use
import pickle
pickle.dump(cv, open("cv_stop.pkl", "wb"))
data_stop.to_pickle("dtm_stop.pkl")


# In[8]:


# Now we will make some wordclouds

from wordcloud import WordCloud

wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)
data.keys()


# In[31]:


# Reset the output dimensions
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 6]

song_names = ['327', 'Block Party', 'Dangerookipawaa freestyle','Leader of Delinquents', 'Shape of you']

# Create subplots for each comedian
for index, song in enumerate(data.columns):
    wc.generate(data_clean.Lyrics[song])
    
    plt.subplot(3, 4, index+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(song_names[index])
    
plt.show()

Findings:
Shape of you has no swear words but other four songs use swear words a lot 
3 of the 4 hiphop songs use their song title a lot in their lyrics
Shape of you  and dangerookipawa freestyle does not use the song title that often in its lyrics

# # 2. Level of Profanity in the Lyrics

# In[10]:


# Find the count of use of profanity in the lyrics

Counter(words).most_common()


# In[11]:


# Now we isolate just those swear words
data_bad_words = data.transpose()[['shit', 'nigga', 'bitch','fuck','fuckin']]
data_profanity = pd.concat([data_bad_words.shit+data_bad_words.fuck+data_bad_words.fuckin,data_bad_words.nigga+ data_bad_words.bitch], axis=1)
data_profanity.columns = ['derogatory_word', 'curse_word']
data_profanity

##derogatory= insulting a group of people'''
##curse=expression of anger'''


# In[14]:


# We will display our findings in a scatterplot
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [15, 5]
song_names = ['327', 'Block Party', 'Dangerookipawaa freestyle', 'Leader of Delinquents', 'Shape of you']


for i, word in enumerate(data_profanity.index):
    x = data_profanity.derogatory_word.loc[word]
    y = data_profanity.curse_word.loc[word]
    plt.scatter(x, y, color='blue')
    plt.text(x-4, y+0.1, song_names[i], fontsize=6)
    plt.xlim(-20, 40) 
    
plt.title('Number of profane words used in Songs', fontsize=20)
plt.xlabel('Number of Derogatory Words', fontsize=15)
plt.ylabel('Number of Curse Words', fontsize=15)

plt.show()

Findings: 
1. Block Party and Dangerookipawaa freestyle are extremes of the dataset. They both strech the data in opposite directions, hence the data is a good collection.
2. Popular Hiphop songs of 2020 tend to have more profanity than earlier pop songs
3. There are two kinds of profanity that hiphop songs can follow, either be more derogatory like Block Party or be more cursing like Dangerookipawaa freestyle 
4. More Pop songs are needed to draw a conclsive finding