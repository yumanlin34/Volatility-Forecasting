# Volatility-Forecasting

The forecast volatility of an asset is one of the most important quantities that portfolio managers use in markets. 
It is used in everything from portfolio optimization to option pricing and risk management. 
Over the years academics and practitioners have developed and used multiple successful techniques for forecasting volatility, 
from very simple statistical estimators to deep learning methods. The topic of volatility forecasting in markets is so important 
that Robert Engle was awarded the Nobel Memorial Prize in Economics in 2003 for his work on volatility prediction using time series methods, 
and in 1997 Myron Scholes and Robert C. Merton were awarded the same prize for their work on option pricing which puts the volatility of an asset 
as the most important factor in determining the value of an option (conversely, the model can be used to determine the market implied forecast volatility 
of an asset from the options market price).

Realized volatility which will be compared against your forecast will be calculated as the square root of the sum of the squared daily returns of the SPY ETF 
and then annualized by multiplying this number by the square root of 52.

Models:
ARCH
GARCH
LSTM
