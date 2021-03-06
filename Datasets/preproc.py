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

test_x_ = pd.read_csv('test-x.csv')
test_x_ = pd.get_dummies(test_x_)
test_x = test_x_[cols_fin[:-1]].values
test_y = test_x_[cols_fin[-1]].values

zeros = [0]*(len(cols_fin) - 1)


index = []
index_ = np.arange(0, train_x.shape[0])
for i in range (1, train_x.shape[0]):
    if i < 5:
        index.append(index_[:i])
    else:
        index.append(index_[(i-5):i])
n_batches = len(index)

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
        train_y_batches.append([train_y[batch[-1]] + 1])
    else:
        train_x_batches.append(train_x[batch, :].flatten().tolist())
        train_y_batches.append([train_y[batch[-1]] + 1])

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
        test_y_batches.append([test_y[batch[-1] + 1]])
    else:
        test_x_batches.append(test_x[batch, :].flatten().tolist())
        test_y_batches.append([test_y[batch[-1] + 1]])

full_d = np.append(train_x_batches, test_x_batches)
full_d_x = np.append(train_y_batches, test_y_batches)

print(full_d.shape)

index = np.arange(full_d.shape[0])
np.random.shuffle(index)

train_index = index[:((4 * len(index)) // 5)]
test_index = index[((4 * len(index)) // 5):]

final_train_x = full_d[train_index, :]
final_train_y = full_d_x[train_index]

final_test_x = full_d[test_index, :]
final_test_y = full_d_x[test_index]

with open('train_x.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerows(final_train_x)

with open('train_y.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerows(final_train_y)

with open('test_x.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerows(final_test_x)

with open('test_y.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerows(final_test_y)