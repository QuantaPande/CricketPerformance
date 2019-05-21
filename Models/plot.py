import numpy as np
import matplotlib.pyplot as plt

weights = np.load('W_shuffle.npy')
x_train = np.genfromtxt('train_x.csv', delimiter = ',')
y_train = np.genfromtxt('train_y.csv')

x = [np.arange((0, 450, 0.1))]*5
x = np.asarray(x)

y_hat_train = weights[0] + np.matmul(x_train[:, [4, 21, 38, 55, 72]], weights[[4, 21, 38, 55, 72]])
y_hat = weights[0] + np.matmul(x, weights[[4, 21, 38, 55, 72]])

plt.plot(np.sum(x, axis = 1), y_hat, label = 'Predicted Line')
plt.scatter(np.sum(x_train[:, [4, 21, 38, 55, 72]], axis = 1), y_hat_train, 'rx', label = 'True Points')
plt.xlabel('Sum of runs scored by the team')
plt.xlabel('Score')

plt.show()

