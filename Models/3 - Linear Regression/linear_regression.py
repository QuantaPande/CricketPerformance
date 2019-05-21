# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 06:59:22 2018

@author: chint
"""

import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = pandas.read_csv('train_india_x.csv', header=None).values
y = pandas.read_csv('train_india_y.csv', header=None).values

X_test = pandas.read_csv('test_india_x.csv', header=None).values
y_test = pandas.read_csv('test_india_y.csv', header=None).values
b = np.ones((X.shape[0], 1))
bt = np.ones((X_test.shape[0], 1))
X = np.concatenate((b,X), axis=1)
X_test = np.concatenate((bt,X_test), axis=1)
acc = []
vacc = []
w = []
l = []
X, X_val, y, y_val = train_test_split(X, y)
for lamda in np.logspace(-4, 4, 100):
    
    W = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)+ lamda*np.eye(X.shape[1])), X.T), y)
    yhat = np.dot(X, W)
    vyhat = np.dot(X_val, W)
    acc.append(1/X.shape[0] * np.sum((y - yhat)**2))
    vacc.append(1/X_val.shape[0] * np.sum((y_val - vyhat)**2))
    w.append(W)
    l.append(lamda)
    
i = np.argmin(vacc)
yhat = np.dot(X, w[i])
yhat_test = np.dot(X_test, w[i])

print('lambda = ', l[i])
print(1/X.shape[0] * np.sum((y - yhat)**2))
print(1/X_test.shape[0] * np.sum((y_test - yhat_test)**2))