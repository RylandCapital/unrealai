import os 
import requests
import pandas as pd
import numpy as np
import norgatedata
import datetime as dt

from dateutil.relativedelta import relativedelta
from scipy.signal import argrelextrema
from scipy import stats
from dotenv import load_dotenv

load_dotenv()
KEOD = os.getenv("EOD")




def get_daily_equity(ticker):
    
    start_date = pd.Timestamp('2010-01-01')
    padding_setting = norgatedata.PaddingType.NONE  

    fundamental_symbols = ["#SPXADR", "#NDXADR", "#SPXMCOSC", "#NDXMCOSC", "#NDXZWBT", "#SPXZWBT",
                           '#OEX%MA50', '#OEX%MA200','#M2FED3','#M2FED2','#CBOEPCE',"$VIX"]
    
    tmpdfs = []
    for i in fundamental_symbols:
        # Retrieve the data as a Pandas DataFrame
        line = norgatedata.price_timeseries(
        i,
        stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.TOTALRETURN,
        padding_setting=padding_setting,
        start_date=start_date,
        timeseriesformat='pandas-dataframe'
        )

        tmpdf = line[['Close']].rename(columns={'Close': i+"{0}".format('_close')})
        tmpdfs.append(tmpdf)

        # tmpdf.plot()

    df_index = pd.concat(tmpdfs, axis=1)


    priceadjust = norgatedata.StockPriceAdjustmentType.TOTALRETURN 
    timeseriesformat = 'pandas-dataframe'

    
    # 2) Pull data for the TICKER
    ts = norgatedata.price_timeseries(
                ticker,
                stock_price_adjustment_setting = priceadjust,
                padding_setting = padding_setting,
                start_date = start_date,
                timeseriesformat=timeseriesformat,
            )

    df = pd.DataFrame.from_dict(ts).reset_index()
    
    df['adjusted_close'] = df['Close'].copy()

    
    # 21-day SMA binary indicator: 1 if price above 21-day SMA
    df['21_day_sma'] = df['adjusted_close'].rolling(window=21).mean()
    df['above_21_sma'] = np.where(df['adjusted_close'] > df['21_day_sma'], 1, 0)
    
    # 55-day SMA binary indicator: 1 if price above 55-day SMA
    df['55_day_sma'] = df['adjusted_close'].rolling(window=55).mean()
    df['above_55_sma'] = np.where(df['adjusted_close'] > df['55_day_sma'], 1, 0)

    # 55-day SMA binary indicator: 1 if price above 200-day SMA
    df['200_day_sma'] = df['adjusted_close'].rolling(window=200).mean()
    df['above_200_sma'] = np.where(df['adjusted_close'] > df['200_day_sma'], 1, 0)
    
    # 21-day RSI calculation and binary conversion (threshold 50)
    delta = df['adjusted_close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=21).mean()
    avg_loss = loss.rolling(window=21).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1.0 + rs))
    df['21day_rsi'] = np.where(rsi > 70, 1, np.where(rsi<30, -1, 0))
    
    # Convert 'date' to datetime if needed
    df['Date'] = pd.to_datetime(df['Date'])
    
    priceadjust = norgatedata.StockPriceAdjustmentType.TOTALRETURN 
    padding_setting = norgatedata.PaddingType.NONE   
    timeseriesformat = 'pandas-dataframe'
    
    # 2) Pull data for the TICKER
    ts = norgatedata.price_timeseries(
                'SPY',
                stock_price_adjustment_setting = priceadjust,
                padding_setting = padding_setting,
                start_date = start_date,
                timeseriesformat=timeseriesformat,
            )

    df_spy = pd.DataFrame.from_dict(ts).reset_index()
    
    df_spy['adjusted_close'] = df_spy['Close'].copy()
    
    
    df_spy['21_day_sma'] = df_spy['adjusted_close'].rolling(window=21).mean()
    df_spy['above_21_sma'] = np.where(df_spy['adjusted_close'] > df_spy['21_day_sma'], 1, 0)
    
    df_spy['55_day_sma'] = df_spy['adjusted_close'].rolling(window=55).mean()
    df_spy['above_55_sma'] = np.where(df_spy['adjusted_close'] > df_spy['55_day_sma'], 1, 0)

    df_spy['200_day_sma'] = df_spy['adjusted_close'].rolling(window=200).mean()
    df_spy['above_200_sma'] = np.where(df_spy['adjusted_close'] > df_spy['200_day_sma'], 1, 0)
    
    delta_spy = df_spy['adjusted_close'].diff()
    gain_spy = delta_spy.clip(lower=0)
    loss_spy = -1 * delta_spy.clip(upper=0)
    avg_gain_spy = gain_spy.rolling(window=21).mean()
    avg_loss_spy = loss_spy.rolling(window=21).mean()
    rs_spy = avg_gain_spy / avg_loss_spy
    rsi_spy = 100 - (100 / (1.0 + rs_spy))
    df_spy['21day_rsi'] = np.where(rsi_spy > 70, 1, np.where(rsi_spy<30, -1, 0))
    
    df_spy['Date'] = pd.to_datetime(df_spy['Date'])
    
    # Drop initial NaNs from rolling calculations in SPY data
    df_spy.dropna(inplace=True)
    
    # 5) Merge Ticker with SPY on date (SPY columns will get the suffix _spy)
    df.dropna(inplace=True)
    # Merge df with df_spy on 'Date'
    merged_df = pd.merge(
        df,
        df_spy[['Date', 'above_21_sma', 'above_55_sma', 'above_200_sma','21day_rsi']],
        on='Date',
        how='inner',
        suffixes=('', '_spy')
    )

    # Merge the resulting DataFrame with df_index on 'Date'
    df_merged = pd.merge(
        merged_df,
        df_index,
        on='Date',
        how='inner'
    )
    


    return df_merged


def momentum_score(ts, days):
    """
    Input: Price time series.
    Output: Annualized exponential regression slope multiplied by the R2.
    """
    x = np.arange(len(ts))
    log_ts = np.log(ts)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)
    annualized_slope = (np.exp(slope) ** 252 - 1) * 100
    score = annualized_slope * (r_value ** 2)
    return score


tickers = ['CVX', 'HON', 'GOOGL', 'META', 'PG', 'JPM', 'ALL', 'GS', 'VZ', 
           'LRCX', 'NVDA', 'BA', 'MCK', 'TMUS', 'EA', 'DECK', 'NFLX', 'AMZN', 
           'MS', 'WMT', 'TSM', 'ORCL', 'IBM', 'QCOM', 'CSCO', 'CMCSA', 
           'AMZN', 'MS', 'WMT', 'TSM', 'ORCL', 'IBM', 'QCOM', 'CSCO', 'CMCSA']

for f in tickers:
    df = get_daily_equity(f)
    df1 = df.iloc[0:int(len(df) - 252)]
    df2 = df.iloc[int(len(df) - 252):]
    df1.to_csv(r'P:\10_CWP Trade Department\Ryland\unrealai\unrealai\traindata\{0}.csv'.format(f))
    df2.to_csv(r'P:\10_CWP Trade Department\Ryland\unrealai\unrealai\testdata\{0}.csv'.format(f))
    df1.set_index('Date')[['Close']].plot(title=f)
    df2.set_index('Date')[['Close']].plot(title=f)



# self.feature_cols = [
#             'above_21_sma',
#             'above_55_sma',
#             'above_200_sma',
#             '21day_rsi',
#             'above_21_sma_spy',
#             'above_55_sma_spy',
#             'above_200_sma_spy',
#             '21day_rsi_spy',
#             "#SPXADR_close",
#             "#NDXADR_close",
#             "#SPXMCOSC_close",
#             "#NDXMCOSC_close",
#             "#NDXZWBT_close",
#             "#SPXZWBT_close",
#             "#OEX%MA50_close",
#             "#OEX%MA200_close",
#             "#M2FED3_close",
#             "#M2FED2_close"
#         ]

