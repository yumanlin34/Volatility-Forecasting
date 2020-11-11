# GARCH(1, 1)
# Algo Depth, Volatility prediction project
# Yuman Lin

import pandas as pd
from arch import arch_model
import datetime as dt
import numpy as np
import scipy
from scipy.stats import norm
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import statsmodels.stats as sms
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from multiprocessing import Pool

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tiingo import TiingoClient

# setting
config = {}

# To reuse the same HTTP Session across API calls (and have better performance), include a session key.
config['session'] = True

# If you don't have your API key as an environment variable,
# pass it in via a configuration dictionary.
config['api_key'] = "b4b0471cd06afd66556f600e6bc3fcc25d91fd4b"

# Initialize
client = TiingoClient(config)

# get the ETF data, only adujusted close price
def get_data(ticker, start = '2005-01-01', end = dt.datetime.now()):
    metric_sep_data = []
    
    metrics = ['adjClose']
    for metric in metrics:
        df = client.get_dataframe(ticker,startDate=start,endDate=end,metric_name=metric).stack()
        df.index = df.index.rename(['date','ticker'])
        df = df.reorder_levels(['ticker','date']).sort_index().rename(metric)
        metric_sep_data.append(df)
    
    res = pd.concat(metric_sep_data,axis=1).reset_index()
    res['date'] = pd.to_datetime(res['date']).dt.date
    
    res.set_index('date', inplace = True)
    res.rename(columns={"ticker": "Ticker", "adjClose": "Adj Close"}, inplace = True)
    res = res.drop('Ticker', axis = 1)
    return res

# caluclate r2
def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2

def GARCH(ticker):
    data = get_data(ticker)
    data['return'] = data['Adj Close'].pct_change()
    data['volatility'] = data['return'].rolling(21).std().shift(-21)
    
    y_true = []
    y_pred = []
    rtn = []
    date_index = []
    
    for i in range(1300, data.shape[0], 21):
        train_set = data.iloc[i-1281: i-21]
        test_set = data.iloc[i]
        
        # model: GARCH(1,1)
        basic_gm = arch_model(train_set['return'], p = 1, q = 1,mean = 'constant', vol = 'GARCH', dist = 'normal')
        # Fit the model
        gm_result = basic_gm.fit()
        # Make 5-period ahead forecast
        gm_forecast = gm_result.forecast(horizon = 21)
        predict_value = np.sqrt(gm_forecast.variance[-1:]["h.21"].values[0])
        
        y_true.append(test_set['volatility'])
        y_pred.append(predict_value)
        date_index.append(data.index[i])
        rtn.append(data['return'][i])
   
    data = {'monthly_return': rtn, 'Predict': y_pred, 'True_value': y_true}
    res = pd.DataFrame(data, index = date_index)
    res.dropna(inplace = True)
    return res

def evaluation_metric(res):
	rmse = mean_squared_error(res['Predict'], res['True_value'])**0.5
	mae = mean_absolute_error(res['Predict'], res['True_value'])
	r2 = rsquared(res['Predict'], res['True_value'])

	res['Error'] = abs(res["Predict"]-res["True_value"])
	res['Precentage_error'] = abs(res["Predict"]-res["True_value"])/res["True_value"]

	print("The RMSE of the model is:", rmse)
	print("The MAE of the model is", mae)
	print("The R_square of the model is", r2)
	plt.figure(figsize=(20,10))
    
	plt.subplot(221)
    
	plt.plot(np.array(res.index), res["Predict"], label = 'Predict')
	plt.plot(np.array(res.index), res["True_value"], label = 'True')
	plt.title('GARCH')
	plt.legend()

	plt.subplot(222)
	plt.plot(np.array(res.index), res['Error'], label = 'Error')
	plt.title('GARCH error')

	plt.subplot(212)
	plt.plot(np.array(res.index), res['Precentage_error'], label = 'Percentage Error')
	#plt.title('GARCH precentage error', loc='left')
	plt.legend()

	return rmse, mae, r2

res = GARCH(['spy'])
print(res.head())

rmse, mae, r2 = evaluation_metric(res)










