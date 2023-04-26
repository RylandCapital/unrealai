import os 
import requests
import pandas as pd
import numpy as np
import time
import datetime as dt

from scipy import stats
from dotenv import load_dotenv

load_dotenv()
#fantasy labs username
KEOD = os.getenv("EOD")


#start and stop '%Y-%m-%d' = 2022-12-30
def get_intra_equity(symbol, start, stop, interval):
        
        start = int(time.mktime(time.strptime(start, '%Y-%m-%d')))
        stop = int(time.mktime(time.strptime(stop, '%Y-%m-%d')))
        
        req = requests.get('https://eodhistoricaldata.com/api/intraday'+ \
                            '/{0}.US?api_token={1}&interval={2}&fmt=json'.format(symbol, KEOD, interval)+ \
                            '&from={0}&to={1}'.format(start,stop))
        df= pd.DataFrame.from_dict(req.json())
        df['timestamp'] = df['timestamp'].apply(
                lambda x: dt.datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')
                )

        return df

def momentum_score(ts):
    """
    Input:  Price time series.
    Output: Annualized exponential regression slope, 
            multiplied by the R2
    """
    # Make a list of consecutive numbers
    x = np.arange(len(ts)) 
    # Get logs
    log_ts = np.log(ts) 
    # Calculate regression values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)
    # Annualize percent
    annualized_slope = (np.power(np.exp(slope), 252) - 1) * 100
    #Adjust for fitness
    score = annualized_slope * (r_value ** 2)
    return score