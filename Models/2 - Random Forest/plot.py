import numpy as np
import matplotlib.pyplot as plt

weights = np.load('W_shuffle.npy')
x_train = np.genfromtxt('train_x.csv', delimiter = ',')
y_train = np.genfromtxt('train_y.csv')

x = [np.arange(-2, 2, 0.001)]*5
x = np.asarray(x)

y_hat_train = weights[0] + np.matmul(x_train[:, [4, 21, 38, 55, 72]], weights[[4, 21, 38, 55, 72]])

print(np.sum(x_train[:, [14, 31, 48, 65, 82]], axis = 1).shape)
y_hat = weights[0] + np.matmul(x.T, weights[[4, 21, 38, 55, 72]])
y_hat = np.squeeze(y_hat)
y_hat_train = np.squeeze(y_hat_train)
print(y_hat_train.shape)

plt.plot(np.mean(x, axis = 0), y_hat, label = 'Predicted Line')
plt.scatter(np.mean(x_train[:, [14, 31, 48, 65, 82]], axis = 1), y_train, c = 'r', marker = 'x', label = 'True Points')
plt.xlabel('Sum of runs scored by the team')
plt.xlabel('Score')

plt.show()

