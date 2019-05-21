import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

n_instances = 10

x_train_ = pd.read_csv("train_x.csv", delimiter = ',', header = None)
x_test_ = pd.read_csv("test_x.csv", delimiter = ',', header = None)
y_train_ = pd.read_csv("train_y.csv", delimiter = ',', header = None)
y_test_ = pd.read_csv("test_y.csv", delimiter = ',', header = None)

x_train = x_train_.values
x_test = x_test_.values

y_train = np.ravel(y_train_.values)
y_test = np.ravel(y_test_.values)

kf = KFold(n_splits = 5, shuffle = True)


index = np.arange(0, x_train.shape[0])
error_test = 0
error_train = 0
B = np.arange(10, 1000, 10)
for i in range(0, 1):
        bag = resample(index, n_samples = np.ceil(x_train.shape[0]/3).astype(int))
        kf.get_n_splits()
        for train_index, test_index in kf.split(x_train[bag, :], y_train[bag]):
                rfclassifier = RandomForestRegressor(n_estimators = 570, bootstrap = True, max_features = 8)
                rfclassifier.fit(x_train[train_index, :], y_train[train_index])
                yhat_test = rfclassifier.predict(x_test)
                error_test = mean_squared_error(y_test, yhat_test)
                yhat = rfclassifier.predict(x_train[test_index, :])
                error_train += mean_squared_error(y_train[test_index], yhat)
        # error_test[i, int(item / 10 - 1)] = error_test[i, int(item / 10 - 1)]/5
        # error_train[i, int(item / 10 - 1)] = error_train[i, int(item / 10 - 1)]/5

y_hat_train = rfclassifier.predict(x_train)
plt.scatter(np.sum(x_train[:, [4, 21, 38, 55, 72]], axis = 1), y_hat_train, label = 'Predicted')
plt.scatter(np.sum(x_train[:, [4, 21, 38, 55, 72]], axis = 1), y_train, label = 'True')
plt.legend()
print(y_hat_train)
plt.show()

print(error_train/5)
print(error_test)

# mean_train = np.mean(error_train, axis = 0)
# mean_test = np.mean(error_test, axis = 0)
# std_test = np.std(error_test, axis = 0)

x = np.arange(10, 1000, 10)

# plt.figure(1)
# plt.plot(x, mean_train, 'rx', linestyle = '-', label = "Mean error on Training Data")
# plt.plot(x, mean_test, 'bo', linestyle = '-', label = "Mean Error on Testing Data")
# plt.plot(x, std_test, 'k-', linestyle = '-', label = "STD DEV on Testing Data")
# plt.xlabel("Number of trees")
# plt.title("Performance of Random Forest Classifier with changing number of Trees")
# plt.legend(loc = "upper right")
# plt.show()

# print(mean_train)
# print(mean_test)
# print(std_test)