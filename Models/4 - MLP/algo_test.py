import numpy as np
import os as os
import backprop as bp
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

train = np.genfromtxt("train_x.csv", delimiter = ",")
test = np.genfromtxt("test_x.csv", delimiter = ",")
train_label = np.genfromtxt("train_y.csv", delimiter = ',')
train_label = np.expand_dims(train_label, axis = 1)
test_label = np.genfromtxt("test_y.csv", delimiter = ',')
test_label = np.expand_dims(test_label, axis = 1)
# print(train_label.shape)

# print("Loading finished")

model = bp.backprop(train, train_label, 2, 50, max_iter = 250, eta = 0.0002, f_type = "R", b_type = "R", b = 16, disp_step = 10)
# model.fit()
# model.baseline(test, test_label)
err = model.score(test, test_label)

# x = [np.arange(-2, 2, 0.001)] * 

y_hat_train = model.predict(train)
plt.scatter(np.sum(train[:, [4, 21, 38, 55, 72]], axis = 1), y_hat_train, label = 'Predicted')
plt.scatter(np.sum(train[:, [4, 21, 38, 55, 72]], axis = 1), train_label, label = 'True')
plt.legend()
print(y_hat_train)
plt.show()

print(err)