import os
import time
import pandas as pd
import numpy as np
import tweepy

from dotenv import load_dotenv

load_dotenv()

# Replace 'twitter' with any Twitter user's screen_name
#tweets = api.user_timeline(screen_name='twitter', count=50)

def twitter_sample(count=100, loops=1):

    # Assign your own Twitter API credentials
    consumer_key = os.getenv('TWITTER_API_KEY')
    consumer_secret = os.getenv('TWITTER_SECRET')
    access_token = os.getenv('TWITTER_ACCESS_TOKEN')
    access_token_secret = os.getenv('TWITTER_ACCESS_SECRET')

    # Authenticate to Twitter
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    df = pd.DataFrame([], columns=['created_at','text'])

    for i in np.arange(loops):
        # Create API object
        api = tweepy.API(auth)
        # Define the search term
        search_words = "a"

        # Collect tweets
        tweets = api.search_tweets(q=search_words, lang='en', count=count)

        # Iterate on tweets
        for tweet in tweets:
            df.loc[tweet._json['id'], 'created_at'] = tweet._json['created_at']
            df.loc[tweet._json['id'], 'text'] = tweet._json['text']
        
        time.sleep(5)

    return df.drop_duplicates('text')
