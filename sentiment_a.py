# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 08:17:36 2019

@author: KJee
"""

import pandas as pd 
import datetime as dt 
from twitterscraper import query_tweets
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langdetect import detect 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.mlab as mlab

def detector(x):
    try:
       return detect(x)
    except:
        None 
        
analyzer = SentimentIntensityAnalyzer()

"""
begin_date = dt.date(2019,3,3)
end_date = dt.date(2019,3,7)

begin_date_premier = dt.date(2019,3,7)
end_date_premier = dt.date(2019,3,9)

tweets_before = query_tweets("#CaptainMarvel", begindate = begin_date, enddate= end_date, limit = 100000)

tweets_after = query_tweets("#CaptainMarvel", begindate = begin_date_premier, enddate = end_date_premier)
                            
df_before = pd.DataFrame(t.__dict__ for t in tweets_before)
df_after = pd.DataFrame(t.__dict__ for t in tweets_after)

#filter for english tweets
df_before['lang'] = df_before['text'].apply(lambda x:detector(x))
df_before = df_before[df_before['lang'] == 'en']
df_after['lang'] = df_after['text'].apply(lambda x: detector(x))
df_after = df_after[df_after['lang'] == 'en'] 

#save files
#df_before.to_csv('cm_tweets_before_clean.csv')
#df_after.to_csv('cm_tweets_after_clean.csv')
"""

df_before = pd.read_csv('cm_tweets_before_clean.csv')
df_after = pd.read_csv('cm_tweets_after_clean.csv')
#get sentiment scores
sentiment_before = df_before['text'].apply(lambda x: analyzer.polarity_scores(x))
sentiment_after = df_after['text'].apply(lambda x: analyzer.polarity_scores(x))

#put sentiment into dataframe
df_before = pd.concat([df_before, sentiment_before.apply(pd.Series)],1)
df_after = pd.concat([df_after, sentiment_after.apply(pd.Series)],1)

#removed duplicates because of sponsored tweets? 
df_before.drop_duplicates(subset = 'text',inplace = True)
df_after.drop_duplicates(subset = 'text',inplace = True)
df_after['timestamp'] = df_after['timestamp'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
df_after = df_after[df_after['timestamp'] > dt.datetime(2019,3,8,0,0,0)]


df_before['compound'].hist()
df_before['compound'].mean()
df_before['compound'].median()

df_after['compound'].hist()
df_after['compound'].mean()
df_after['compound'].median()


before_ratio = df_before[df_before['compound'] > 0].shape[0] / df_before[df_before['compound'] < 0].shape[0]
after_ratio = df_after[df_after['compound'] > 0].shape[0] / df_after[df_after['compound'] < 0].shape[0]

df_before_nz = df_before[df_before['compound'] != 0]
df_after_nz = df_after[df_after['compound'] != 0]

df_before_nz['compound'].sample(5000).hist()
df_after_nz['compound'].sample(5000).hist()

################################################################################### 
"""
begin_date_a = dt.date(2018,4,23)
end_date_a = dt.date(2018,4,27)

begin_date_premier_a = dt.date(2018,4,28)
end_date_premier_a = dt.date(2018,4,30)

a_tweets_before = query_tweets("#Avengers OR #InfinityWar", begindate = begin_date_a, enddate= end_date_a, limit = 200000)
a_tweets_after = query_tweets("#Avengers OR #InfinityWar", begindate = begin_date_premier_a, enddate = end_date_premier_a, limit = 200000)
                            
a_df_before = pd.DataFrame(t.__dict__ for t in a_tweets_before)
a_df_after = pd.DataFrame(t.__dict__ for t in a_tweets_after)

#filter for english tweets
a_df_before['lang'] = a_df_before['text'].apply(lambda x:detector(x))
a_df_before = a_df_before[a_df_before['lang'] == 'en']
a_df_after['lang'] = a_df_after['text'].apply(lambda x: detector(x))
a_df_after = a_df_after[a_df_after['lang'] == 'en']

#save files
#a_df_before.to_csv('avengers_tweets_before_clean.csv')
#a_df_after.to_csv('avengers_tweets_after_clean.csv')
"""

a_df_before = pd.read_csv('avengers_tweets_before_clean.csv', engine = 'python')
a_df_after = pd.read_csv('avengers_tweets_after_clean.csv')

a_df_before.text = a_df_before.text.astype(str)
a_df_after.text = a_df_after.text.astype(str)

#get sentiment scores
a_sentiment_before = a_df_before['text'].apply(lambda x: analyzer.polarity_scores(x))
a_sentiment_after = a_df_after['text'].apply(lambda x: analyzer.polarity_scores(x))

#put sentiment into dataframe
a_df_before = pd.concat([a_df_before, a_sentiment_before.apply(pd.Series)],1)
a_df_after = pd.concat([a_df_after, a_sentiment_after.apply(pd.Series)],1)

#removed duplicates because of sponsored tweets? 
a_df_before.drop_duplicates(subset = 'text',inplace = True)
a_df_after.drop_duplicates(subset = 'text',inplace = True)
#df_after['timestamp'] = df_after['timestamp'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
#df_after = df_after[df_after['timestamp'] > dt.datetime(2019,3,8,0,0,0)]


a_df_before['compound'].hist()
a_df_before['compound'].mean()
a_df_before['compound'].median()

a_df_after['compound'].hist()
a_df_after['compound'].mean()
a_df_after['compound'].median()


a_before_ratio = a_df_before[a_df_before['compound'] > 0].shape[0] / a_df_before[a_df_before['compound'] < 0].shape[0]
a_after_ratio = a_df_after[a_df_after['compound'] > 0].shape[0] / a_df_after[a_df_after['compound'] < 0].shape[0]

a_df_before_nz = a_df_before[a_df_before['compound'] != 0]
a_df_after_nz = a_df_after[a_df_after['compound'] != 0]

a_df_before_nz['compound'].sample(5000).hist()
a_df_after_nz['compound'].sample(5000).hist()



############################################################################### Comparison 

a_df_before_nz['compound'].sample(10000).hist()
a_df_after_nz['compound'].sample(10000).hist()
df_before_nz['compound'].sample(10000).hist()
df_after_nz['compound'].sample(10000).hist()

abmean = a_df_before_nz['compound'].mean()
abstd = a_df_before_nz['compound'].std()
a_df_before_nz['compound'].mean()
a_df_after_nz['compound'].mean()
df_before_nz['compound'].mean()
df_after_nz['compound'].mean()

ax1 = sns.distplot(a_df_before_nz['compound'], bins=15, hist = False, label = 'Avengers Before Premier', color = 'r', kde_kws={'linestyle':'--'})
ax2 = sns.distplot(a_df_after_nz['compound'], bins=15, hist = False, label = 'Avengers After Premier',color= 'r')

ax3 = sns.distplot(df_before_nz['compound'], bins=15, hist = False, label = 'CM Before Premier', color ='blue',  kde_kws={'linestyle':'--'})
ax4 = sns.distplot(df_after_nz['compound'], bins=15, hist = False, label = 'CM After Premier', color ='blue')
plt.legend()
plt.title('Infinity War vs. Captain Marvel Sentiment')


#samples
    
for i in df_after[df_after['compound'] >= .9]['text'].sample(5):
    print(i)
    print(' ')
    
for i in df_after[df_after['compound'] <= -.9]['text'].sample(5):
    print(i)
    print(' ')
