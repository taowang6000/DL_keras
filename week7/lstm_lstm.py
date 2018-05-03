'''
Based on mstep.py of week3, using cyclic validation method
'''

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot
from numpy import array
from numpy import fabs
from numpy import mean
 
# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

def mean_absolute_percentage_error(a, b): 
    mask = a != 0
    return (fabs(a - b)/a)[mask].mean()
 
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
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_val, n_neurons1, n_neurons2):
	# reshape training into [samples, timesteps, features]
	X, y = train[:, 0:n_lag], train[:, n_lag:]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	# design network
	model = Sequential()
	model.add(LSTM(n_neurons1, return_sequences=True, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
	model.add(LSTM(n_neurons2))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', metrics = ['mae'], optimizer='adam')
	# fit network
	all_mae_history = []
	for i in range(len(n_val)):
		# Prepare training data and validation data
		train_X, train_y = X[0:n_val[i], :, :], y[0:n_val[i], :]
		if n_val[i] == 465:
			vali_X, vali_y = X[-22:, :, :], y[-22:, :]
		else:
			vali_X, vali_y = X[n_val[i] : n_val[i]+22, :, :], y[n_val[i] : n_val[i]+22, :]

		turn_mae = []
		for j in range(nb_epoch):
			model.fit(train_X, train_y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
			val_mse, val_mae = model.evaluate(vali_X, vali_y, batch_size=n_batch, verbose = 0)
			turn_mae.append(val_mae)
			model.reset_states()
		all_mae_history.append(turn_mae)
	average_mae_history = [mean([x[k] for x in all_mae_history]) for k in range(nb_epoch)]
	fig = pyplot.figure()
	pyplot.plot(range(1, len(average_mae_history) + 1), average_mae_history)
	pyplot.xlabel('Epochs (LSTM 1 neurons: {0}, LSTM 2 Neurons: {1})'.format(n_neurons1, n_neurons2))
	pyplot.ylabel('Validation Mean Absolute Error')
	#pyplot.show()
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
def evaluate_rmse(test, forecasts, n_lag, n_seq, turn = 1):
	print ('RMSE: ', end = '')
	sum = 0
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		#mape = mean_absolute_percentage_error(actual, predicted)
		print('t+{0}: {1:.1f}, '.format(i+1, rmse), end = '')
		sum = sum + rmse
	print('average: {0:.1f}, turn {1}'.format(sum/7, turn))

def evaluate_mape(test, forecasts, n_lag, n_seq, turn = 1):
	print ('MAPE: ', end = '')
	sum = 0
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		#rmse = sqrt(mean_squared_error(actual, predicted))
		mape = mean_absolute_percentage_error(array(actual), predicted)
		print('t+{0}: {1:.2f}, '.format(i+1, mape), end = '')
		sum = sum + mape
	print('average: {0:.2f}, turn {1}'.format(sum/7, turn))


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


'''
from numpy.random import seed
import tensorflow as tf
import random as rn

import os
os.environ['PYTHONHASHSEED'] = '0'
seed(42)
rn.seed(12345)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K
tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
'''

a = list(range(4,20))
b = list(range(4,20))
for i in a[::4]:
		# load dataset
		series = read_csv('transformed.csv', sep = ',', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
		# configure
		n_lag = 1
		n_seq = 7
		n_test = 22
		n_epochs = 100
		n_batch = 1
		n_val = [409, 437, 465]   # validation turns
		n_neurons1 = i
		n_neurons2 = j
		# prepare data
		scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
		# fit model
		model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_val, n_neurons1, n_neurons2)
pyplot.show()
'''
# make forecasts
forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)
# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test + n_seq - 1)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test + n_seq - 1)
# evaluate forecasts
evaluate_rmse(actual, forecasts, n_lag, n_seq)
evaluate_mape(actual, forecasts, n_lag, n_seq)
'''
#pyplot.show()
# plot forecasts
#plot_forecasts(series, forecasts, n_test + n_seq - 1)
