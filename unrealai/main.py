from utils.openai_interact import ChatApp

import numpy as np
import pandas as pd
import time






df = pd.DataFrame([], columns=['topic','sentiment 1-10','raw_text'])
tweets = twitter_sample(loops=1, count=100)
for i in np.arange(len(tweets)):
    sentiment_chat = ChatApp("You are a sentiment anayalst that ranks tweets between 1 and 10 depending on how positive or negative the tweet is. Please simply reply 'Score: ' with your decided score to all text provided")
    topic_chat = ChatApp("You are a grammar professor who identifies the word that represents the subject of a sentence, if any. Please simply reply 'Subject: ' 1 noun that best represents the subject of the sentence. If there is no decipherable subject then reply 'Subject: None'.")
    df.loc[i,'topic'] = topic_chat.chat(tweets.iloc[i]['text']).content.split('Subject: ')[1]
    df.loc[i,'sentiment 1-10'] = sentiment_chat.chat(tweets.iloc[i]['text']).content.split('Score: ')[1]
    df.loc[i,'raw_text'] = tweets.iloc[i]['text']
    time.sleep(10)


df.to_csv(r'C:\Users\rmathews\Downloads\tweets.csv')



    








