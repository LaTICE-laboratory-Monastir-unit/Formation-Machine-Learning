import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import tensorflow as tf
import pandas as pan
import numpy as np

training_epochs = 250


data = pan.read_csv('data/money.csv')

x_train = data.loc[0:1000, ['X1', 'X2', 'X3', 'X4']].as_matrix()
y_train = data.loc[0:1000, ['Label1', 'Label2']].as_matrix()

x_test = data.loc[1001: len(data), ['X1', 'X2', 'X3', 'X4']].as_matrix()
y_test = data.loc[1001: len(data), ['Label1', 'Label2']].as_matrix()

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.placeholder(tf.float32, shape=[None, 2])

w1 = tf.Variable(tf.zeros([4, 4]))
b1 = tf.Variable(tf.zeros([4]))

w2 = tf.Variable(tf.zeros([4, 2]))
b2 = tf.Variable(tf.zeros([2]))

z1 = tf.matmul(x, w1) + b1
a1 = tf.sigmoid(z1)

z2 = tf.matmul(a1, w2) + b2
a2 = tf.sigmoid(z2)

y_pred = tf.nn.softmax(a2)

learning_rates = [0.01, 0.03, 0.1, 0.3, 1]
num_points = learning_rates * (training_epochs+1)

accuracies = np.array(num_points)
alphas = np.array(num_points)
epochs = np.array(num_points)
k = 0

for learning_rate in learning_rates:
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    for j in range(training_epochs + 1):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for step in range(j):
            _, cost = sess.run([optimizer, cross_entropy], feed_dict={x: x_train, y: y_train})

        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100

        acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})

        print('J=', j, ' {:0.5f}'.format(learning_rate),' : {:05.2f}'.format(acc), '%')

        accuracies[k] = acc
        alphas[k] = learning_rate
        epochs[k] = j
        k = k + 1

        sess.close()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(epochs, alphas, accuracies, cmap='jet')
plt.show()
