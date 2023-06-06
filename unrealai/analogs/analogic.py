import os
import pandas as pd
import numpy as np
import datetime as dt
import requests

from multiprocessing import Pool
from dateutil.relativedelta import relativedelta
from scipy.signal import argrelextrema
from dtaidistance import dtw

from dotenv import load_dotenv

load_dotenv()
#fantasy labs username
EOD = os.getenv("EOD")

def analogic(stock_ticker):

    history_years=10
    projection_timeframe=30
    days_back=500
    analog_days= 90
    

    stop = dt.datetime.today()
    start = stop - relativedelta(years=history_years)

    req = requests.get('https://eodhistoricaldata.com/api/eod' +
                        '/{0}.US?api_token={1}&fmt=json'.format(stock_ticker, EOD) +
                        '&period=d&from={0}&to={1}'.format(start.strftime('%Y-%m-%d'),
                                                            stop.strftime('%Y-%m-%d')
                                                            ))
    df = pd.DataFrame.from_dict(req.json())

    df['Return'] = df['adjusted_close'].pct_change()
    df['LogReturn'] = np.log1p(df['adjusted_close'].pct_change())

    #you now have df - historical data from 10 years ago to TODAY

    entries = []
    for h in np.arange(projection_timeframe, days_back)[::-1]:

        print(h)

        # Mimiced data stopping in past
        histdf = df.iloc[:-h]
        #Get the index of the last day in mimiced current day data
        endindex = histdf.index[-1]
        # Get the log-returns for the last analog_days days this is used as "todays data we are trying to match"
        current_period = histdf.tail(analog_days)['LogReturn'].to_numpy()
        # Get the actual return of from the end of mimiced data 30 days forward 
        actual = df.loc[:endindex+projection_timeframe].tail(projection_timeframe+1)['close']
        current_return30 = actual.iloc[-1]/actual.iloc[0]-1

        # Initialize the top 10 best matches list
        top_matches = pd.DataFrame([], columns=['start','stop','distance'])

        # Loop over historical periods and find the best matches
        for i in range(analog_days, len(histdf) - analog_days):

            historical_period = histdf.iloc[i-analog_days:i][['adjusted_close','LogReturn']]
            distance = dtw.distance(current_period, historical_period['LogReturn'].to_numpy())
            
            top_matches.loc[i,'start'] = i-analog_days
            top_matches.loc[i,'stop'] = i
            top_matches.loc[i,'distance'] = distance

            top_matches.sort_values('distance',inplace=True)
            top_matches.reset_index(drop=True,inplace=True)

        top_ten_filtered = pd.DataFrame([], columns=['start','stop','distance'])
        top_ten_filtered.loc[0,'start'] = top_matches.loc[0,'start']
        top_ten_filtered.loc[0,'stop'] = top_matches.loc[0,'stop']
        top_ten_filtered.loc[0,'distance'] = top_matches.loc[0,'distance']

        count=0
        for i in top_matches.index[1:]:
            if all([(abs((l - top_matches.loc[i,'start']))>20) for l in top_ten_filtered['start']])==True:
                top_ten_filtered.loc[i,'start'] = top_matches.loc[i,'start']
                top_ten_filtered.loc[i,'stop'] = top_matches.loc[i,'stop']
                top_ten_filtered.loc[i,'distance'] = top_matches.loc[i,'distance']
                count+=1
                if count>8:
                    break
        
        analog_perf_df = pd.DataFrame([], columns=['return'])
        for match, ix in zip(top_ten_filtered.index, np.arange(len(top_ten_filtered))):
                #what happened for the 30 days AFTER one of the 
                aftermatch30 = histdf.iloc[top_ten_filtered.loc[match]['stop']:top_ten_filtered.loc[match]['stop']+ \
                                           projection_timeframe].reset_index(drop=True)
                analog_perf_df.loc[ix,'return'] = round(aftermatch30['close'].iloc[-1]/aftermatch30['close'].iloc[0] - 1,4)
        
        entry = analog_perf_df.astype(float).describe().iloc[1:].T
        entry['median'] = analog_perf_df.astype(float).median()
        entry['best'] = analog_perf_df.iloc[0].astype(float)
        entry['actual'] = current_return30
        entries.append(entry)
    
    backtest = pd.concat(entries).reset_index(drop=True)
    backtest['ticker'] = stock_ticker
    backtest.to_csv(r'P:\10_CWP Trade Department\_Matrix_\cwp_charts\analogic\{0}.csv'.format(stock_ticker), index=False)




if __name__ == "__main__":

    edip_allo = 'P:\\10_CWP Trade Department\\Smitty\\DSIP allocation.xlsx' 
    edip = pd.read_excel(edip_allo)[['Unnamed: 1', 'Wgting']]
    edip.columns = ['Symbol', 'Target Allocation']
    edip.dropna(axis=0, inplace=True)
    edip.set_index(['Symbol'], inplace=True)
    edip.sort_values(by='Symbol', inplace=True)

    stocks = edip.index.tolist() + ['SPY', 'QQQ', 'DIA', 'IWF'] 
    pool = Pool(processes=len(stocks))
    pool.map(analogic, stocks)
    pool.close()
    pool.join() 