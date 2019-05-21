# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 05:40:43 2018

@author: chint
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas
import numpy as np
from matplotlib import pyplot as plt

X = pandas.read_csv('train_x.csv', header=None).values
y = pandas.read_csv('train_y.csv', header=None).values

X_test = pandas.read_csv('test_x.csv', header=None).values
y_test = pandas.read_csv('test_y.csv', header=None).values

X = X.reshape((X.shape[0], 1, X.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()
model.add(LSTM(17, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(X, y, epochs=50, batch_size=64, 
                    validation_data=(X_test, y_test), shuffle=False)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

yhat = model.predict(X_test)