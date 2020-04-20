# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 13:51:41 2017

@author: shamsul
"""



############################################################
############### MULTI-STEP ARIMA MODEL #####################
############## DIRECT FORECAST STRATEGY ####################
############################################################

from statsmodels.tsa.stattools import adfuller, acf, pacf
import pandas as pd, numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pylab as plt
import matplotlib
from math import sqrt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
matplotlib.rcParams.update({'font.size': 15})
 
# create a differenced series
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


df = pd.read_csv('N.csv')

df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
ts1 = df.set_index('utc_timestamp')
ts = ts1['load']
ts_week = ts.resample('D').mean()

plt.plot(ts_week) 

series = ts_week

result = adfuller(series)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
 print('\t%s: %.3f' % (key, value))


X = series.values
days_in_year = 365
differenced = difference(X, days_in_year)


result = adfuller(differenced)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
 print('\t%s: %.3f' % (key, value))
# 
 
#ACF and PACF plots

lag_acf = acf(differenced, nlags=10)
lag_pacf = pacf(differenced, nlags=10, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-7.96/np.sqrt(len(differenced)),linestyle='--',color='gray')
plt.axhline(y=7.96/np.sqrt(len(differenced)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-7.96/np.sqrt(len(differenced)),linestyle='--',color='gray')
plt.axhline(y=7.96/np.sqrt(len(differenced)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout() 

#
n = 3 # varies accroding to number of step forecast required
split_point = len(series) - n
dataset, validation = series[0:split_point], series[split_point:]

model = ARIMA(dataset, order=(2,1,2))
model_fit = model.fit(disp=0)
# multi-step out-of-sample forecast
start_index = len(dataset)
end_index = start_index + (n-1)
forecast = model_fit.predict(start=start_index, end=end_index)


# invert the differenced forecast to something usable
history = [x for x in X]
day = 1
for yhat in forecast:
	inverted = inverse_difference(history, yhat, days_in_year)
	print('Day %d: %f' % (day, inverted))
	history.append(inverted)
	day += 1



## step =3
forecast_series = pd.Series([36020.047252,34928.404937,38958.612724])
validation_series = pd.Series([41697.083333,40465.416667,35131.333333])


#create a panda series for forecast & then calculate error.
error = sqrt(mean_squared_error(validation_series, forecast_series))
print('\n')
print('Printing Mean Squared Error of Predictions...')
print('RMSE: %.f' % error)

fig, ax = plt.subplots()
ax.set(title='multi step load forecast', xlabel='Date in days', ylabel='load')
ax.plot(forecast_series,'r', label='forecast') 
ax.plot(validation_series,'b', label='true data')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')


############################################################
############### MULTI-STEP LSTM MODEL ######################
############## DIRECT FORECAST STRATEGY ####################
############################################################



# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 02:14:12 2017

@author: shamsul
"""
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array
import pandas as pd
import matplotlib.pylab as plt



# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
	# transform data to be stationary
	diff_series = difference(raw_values, 1)
	diff_values = diff_series.values
	diff_values = diff_values.reshape(len(diff_values), 1)
	# rescale values to -1, 1
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_values = scaler.fit_transform(diff_values)
	scaled_values = scaled_values.reshape(len(scaled_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(scaled_values, n_lag, n_seq)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return scaler, train, test

# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
	# reshape training into [samples, timesteps, features]
	X, y = train[:, 0:n_lag], train[:, n_lag:]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	# design network
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam')
	# fit network
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
		model.reset_states()
	return model

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]

# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = forecast_lstm(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts

# invert differenced forecast
def inverse_difference(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted

# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# create array from forecast
		forecast = array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]
		# invert differencing
		index = len(series) - n_test + i - 1
		last_ob = series.values[index]
		inv_diff = inverse_difference(last_ob, inv_scale)
		# store
		inverted.append(inv_diff)
	return inverted

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))

# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	pyplot.plot(series.values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i - 1
		off_e = off_s + len(forecasts[i]) + 1
		xaxis = [x for x in range(off_s, off_e)]
		yaxis = [series.values[off_s]] + forecasts[i]
		pyplot.plot(xaxis, yaxis, color='red')
	# show the plot
	pyplot.show()

# load dataset
df = pd.read_csv('N.csv')

df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
ts1 = df.set_index('utc_timestamp')
ts = ts1['load']
ts_week = ts.resample('D').mean()

#plt.plot(ts_week) 

series = ts_week
# configure
n_lag = 1
n_seq = 360
n_test = 3
n_epochs = 1
n_batch = 1
n_neurons = 1
# prepare data
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
# fit model
model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
# make forecasts
forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)
# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test+359)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test+359)
# evaluate forecasts
evaluate_forecasts(actual, forecasts, n_lag, n_seq)
# plot forecasts
plot_forecasts(series, forecasts, n_test+359)

error = sqrt(mean_squared_error(actual, forecasts))

print('\n')
print('Printing Mean Squared Error of Predictions...')
print('Test RMSE: %.f' % error)

forecast_series = pd.Series([41342.281331,43045.693929,42742.555379])

validation_series = pd.Series([40595.208333,41697.083333,40465.416667])
#
fig, ax = plt.subplots()
ax.set(title='multi step load forecast', xlabel='Date in days', ylabel='load')
ax.plot(forecast_series,'r', label='forecast') 
ax.plot(validation_series,'b', label='true data')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')
