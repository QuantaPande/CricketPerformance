import numpy as np
import os as os
import backprop as bp
import tensorflow as tf
import matplotlib.pyplot as plt

def OneHotEncoder(data):
    onehot = np.zeros((data.size, 10))
    for i in range(0, data.size):
        onehot[i, int(data[i])] = 1
    return onehot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

train = np.genfromtxt("mnist_train.csv", delimiter = ",")
test = np.genfromtxt("mnist_test.csv", delimiter = ",")
print("Loading finished")
train_label = train[:, 0]
train_label = OneHotEncoder(train_label)

test_label = test[:, 0]
test_label = OneHotEncoder(test_label)
print('converted labels')
model = bp.backprop(train[:, 1:]/255, train_label, 2, 256, max_iter = 100, eta = 0.002, f_type = "C", b_type = "R", b = 100, disp_step = 10)
model.fit()
model.baseline(test[:, 1:], test_label)
err = model.score(test[:, 1:]/255, test_label)
target_num = np.asarray([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
target_num = np.reshape(target_num, (1, 10))
image = model.predict(target_num, direction = "B")
disp = np.reshape(image, (28, 28))
disp = disp*255
seven = disp.astype(int)
seven[seven > 255] = 255
seven[seven < 0] = 0
print(seven)
plt.imshow(seven)
plt.imsave(os.getcwd() + "/result3.png", seven, cmap = plt.get_cmap('gray'))
print(err)