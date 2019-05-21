import pandas as pd
import numpy as np
import csv

train_x_ = pd.read_csv('train-x.csv')
train_x_ = pd.get_dummies(train_x_)
cols_train = list(train_x_)
cols_fin = cols_train[-4:] + cols_train[:-4]
print(cols_fin)
train_x = train_x_[cols_fin[:-1]].values
train_y = train_x_[cols_fin[-1]].values
print(train_x)

test_x_ = pd.read_csv('test-x.csv')
test_x_ = pd.get_dummies(test_x_)
test_x = test_x_[cols_fin[:-1]].values
test_y = test_x_[cols_fin[-1]].values
print(test_y)

zeros = [0]*(len(cols_fin) - 1)

index = []
index_ = np.arange(0, train_x.shape[0])
for i in range (1, train_x.shape[0]):
    if i < 5:
        index.append(index_[:i])
    else:
        index.append(index_[(i-5):i])
n_batches = len(index)

print(index[1])
print(index[3])

train_x_batches = []
test_x_batches = []
train_y_batches = []
test_y_batches = []

for batch in index:
    if len(batch) < 5:
        dummy = (5 - len(batch))*zeros
        for item in train_x[batch, :].flatten().tolist():
            dummy.append(item)
            
        train_x_batches.append(dummy)
        train_y_batches.append([train_y[batch[-1]]])
    else:
        train_x_batches.append(train_x[batch, :].flatten().tolist())
        train_y_batches.append([train_y[batch[-1]]])

with open('train_x.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerows(train_x_batches)

with open('train_y.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerows(train_y_batches)

index = []
index_ = np.arange(0, test_x.shape[0])
for i in range(1, test_x.shape[0]):
    if i < 5:
        index.append(index_[:i])
    else:
        index.append(index_[(i-5):i])
n_batches = len(index)

for batch in index:
    if len(batch) < 5:
        dummy = (5 - len(batch))*zeros
        for item in test_x[batch, :].flatten().tolist():
            dummy.append(item)
        test_x_batches.append(dummy)
        test_y_batches.append([test_y[batch[-1]]])
    else:
        test_x_batches.append(test_x[batch, :].flatten().tolist())
        test_y_batches.append([test_y[batch[-1]]])

with open('test_x.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerows(test_x_batches)

with open('test_y.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerows(test_y_batches)