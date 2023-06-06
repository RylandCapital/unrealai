import pandas as pd

import os
from os import listdir
from os.path import isfile, join

mypath = r'P:\\10_CWP Trade Department\\_Matrix_\\cwp_charts\\analogic'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
files = [pd.read_csv(mypath + "\\" + i, index_col=10) for i in onlyfiles]
data = pd.concat(files)

data['mean_pct'] = data.groupby(level=0)['mean'].rank(pct=True)
data['median_pct'] = data.groupby(level=0)['median'].rank(pct=True)
data['best_pct'] = data.groupby(level=0)['best'].rank(pct=True)
data['std_pct'] = data.groupby(level=0)['std'].rank(pct=True)

data['sharpe'] = data['median']/data['std']
data['sharpe_pct'] = data.groupby(level=0)['sharpe'].rank(pct=True)

mean_corr = data.groupby(level=0)[['mean','actual']].corr().iloc[0::2,-1].sort_values(ascending=False)
median_corr = data.groupby(level=0)[['median','actual']].corr().iloc[0::2,-1].sort_values(ascending=False)
std_corr = data.groupby(level=0)[['std','actual']].corr().iloc[0::2,-1].sort_values(ascending=False)
sharpe_corr = data.groupby(level=0)[['sharpe','actual']].corr().iloc[0::2,-1].sort_values(ascending=False)
best_corr = data.groupby(level=0)[['best','actual']].corr().iloc[0::2,-1].sort_values(ascending=False)

data[['mean_pct','actual']].plot.scatter(x='actual', y='mean_pct')

bottomdf = data[(data['std_pct']<=.10)]
topdf = data[(data['std_pct']>=.90)]

bottommean = bottomdf.groupby(level=0)[['actual']].median()
topmean = topdf.groupby(level=0)[['actual']].median()
compare_mean = pd.concat([bottommean, topmean], axis=1)
compare_mean.columns = ['20th percentile and lower', '80th percentile and higher']
compare_mean['diff'] = compare_mean['80th percentile and higher'] - compare_mean['20th percentile and lower']
compare_mean.sort_values('diff')

compare_mean['diff'].median()


