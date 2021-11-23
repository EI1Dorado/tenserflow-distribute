# # import matplotlib.pyplot as plt
# # import tensorflow as tf
# # import numpy as np
# #
# #
# # def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
# #     layer_name = "layer%s" % n_layer
# #     with tf.name_scope(layer_name):
# #         with tf.name_scope("Weights"):
# #             Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
# #             #概率分布的形式
# #             tf.summary.histogram(layer_name+'/weights',Weights)
# #         with tf.name_scope("biases"):
# #             biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
# #             tf.summary.histogram(layer_name + '/biases', biases)
# #         with tf.name_scope("Wx_plus_b"):
# #             Wx_plus_b = tf.add(tf.matmul(inputs,Weights),biases)
# #         if activation_function is None:
# #             outputs = Wx_plus_b
# #         else:
# #             outputs = activation_function(Wx_plus_b)
# #         return outputs
# #
# #
# # x_data = np.linspace(-1,1,300)[:,np.newaxis]
# # noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
# # y_data = np.square(x_data) - 0.5 + noise
# #
# # #None表示给多少个sample都可以
# # with tf.name_scope("input"):
# #     xs = tf.placeholder(tf.float32,[None,1],name='x_input')
# #     ys = tf.placeholder(tf.float32,[None,1],name='y_input')
# #
# # l1 = add_layer(xs,1,10,1,activation_function=tf.nn.relu)
# # prediction = add_layer(l1,10,1,2,activation_function=None)
# #
# # with tf.name_scope('loss'):
# #     loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
# #                          reduction_indices=[1]))
# #     tf.summary.scalar("loss",loss)
# #
# #
# # with tf.name_scope('train'):
# #     train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
# #
# #
# # init = tf.global_variables_initializer()
# #
# # with tf.Session() as sess:
# #     # 1.2之前 tf.train.SummaryWriter("logs/",sess.graph)
# #     merged = tf.summary.merge_all()
# #     writer = tf.summary.FileWriter('logs/',sess.graph)
# #     sess.run(init)
# #     for i in range(1000):
# #         sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
# #         if i % 50 == 0:
# #             result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
# #             writer.add_summary(result,i)
# # fig = plt.figure(figsize=(18,6),dpi=1600)
# # alpha = alpha_scatterplot=.2
# # alpha_bar_chart = .55
# # fig = plt.figure()
# # ax = fig.add_subplot(333)
# # merged =tf.train.SummaryWritter()
# from keras.datasets import mnist
# import numpy as np
# (train_images,train_labels),(test_images,test_labels)=mnist.load_data()
# test_images.shape
# len(train_labels)
# train_labels
#
# x=np.random.random((64,3,32,10))
# y=np.random.random((32,10))
# z=np.maximum(x,y)
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
#
#
# lr= 0.001
# training_iters = 1000000
# batch_size = 128
#
# n_inputs = 28
# n_steps = 28
# n_hidden_units = 128
# n_classes = 10
#
#
#
# xs = tf.placeholder(tf.float32,[None,n_inputs,n_steps])
# ys = tf.placeholder(tf.float32,[None,10])
#
#
# weights = {
#     'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
#     'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
# }
#
# biases = {
#     'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
#     'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
# }
#
#
# def RNN(X,weights,biases):
#     # hidden layer for input to cell
#     # X(128,28,28)
#     X = tf.reshape(X,[-1,n_inputs])
#     X_in = tf.matmul(X,weights['in']) + biases['in']
#     X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])
#
#     lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
#     #这里生成的state是tuple类型的，因为声明了state_is_tuple参数
#     init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
#     #time_major指时间点是不是在主要的维度，因为我们的num_steps在次维，所以定义为了false
#     outputs,final_states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=init_state,time_major=False)
#
#     #final_states[1] 就是短时记忆h
#     results = tf.matmul(final_states[1],weights['out']) + biases['out']
#
#     return results
#
# prediction = RNN(xs,weights,biases)
#
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=ys))
#
# train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
#
# correct_pred = tf.equal(tf.argmax(prediction,axis=1),tf.argmax(ys,axis=1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#
#     sess.run(init)
#     step = 0
#     while step * batch_size < training_iters:
#         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#         #一个step是一行
#         batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
#         sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
#         if step % 20 == 0:
#             print(sess.run(accuracy, feed_dict={xs: batch_xs, ys: batch_ys}))
#
#         step = step + 1

# -*- coding:utf-8 -*-
# import numpy as np
#
# np.random.seed(123)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Embedding, LSTM
from keras.utils import np_utils
from keras.datasets import mnist

from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import Adam

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


# 载入mnist数据集(第一次执行需要下载数据)
def loda_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # 5. Preprocess input data
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    print
    X_train.shape, Y_train.shape
    print
    X_test.shape, Y_test.shape
    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = loda_mnist();
    model = Sequential()
    # 2个卷积层
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))  # 第一层卷积
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 第一层池化
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(14, 14, 1)))  # 第二层卷积
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 第二层池化
    model.add(Dropout(0.25))  # 添加节点keep_prob
    # 2个全连接层
    model.add(Flatten())  # 将多维数据压成1维，方便全连接层操作
    model.add(Dense(1024, activation='relu'))  # 添加全连接层
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 训练模型
    model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)
    # 评估模型
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)












# import pandas as pd
# from datetime import datetime
# from matplotlib import pyplot
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from numpy import concatenate
# from math import sqrt
#
# # load data
# def parse(x):
#     return datetime.strptime(x, '%Y %m %d %H')
#
#
# def read_raw():
#     dataset = pd.read_csv('raw.csv', parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
#     dataset.drop('No', axis=1, inplace=True)
#     # manually specify column names
#     dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
#     dataset.index.name = 'date'
#     # mark all NA values with 0
#     dataset['pollution'].fillna(0, inplace=True)
#     # drop the first 24 hours
#     dataset = dataset[24:]
#     # summarize first 5 rows
#     print(dataset.head(5))
#     # save to file
#     dataset.to_csv('pollution.csv')
#
#
# def drow_pollution():
#     dataset = pd.read_csv('pollution.csv', header=0, index_col=0)
#     values = dataset.values
#     # specify columns to plot
#     groups = [0, 1, 2, 3, 5, 6, 7]
#     i = 1
#     # plot each column
#     pyplot.figure(figsize=(10, 10))
#     for group in groups:
#         pyplot.subplot(len(groups), 1, i)
#         pyplot.plot(values[:, group])
#         pyplot.title(dataset.columns[group], y=0.5, loc='right')
#         i += 1
#     pyplot.show()
#
#
# def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
#     # convert series to supervised learning
#     n_vars = 1 if type(data) is list else data.shape[1]
#     df = pd.DataFrame(data)
#     cols, names = list(), list()
#     # input sequence (t-n, ... t-1)
#     for i in range(n_in, 0, -1):
#         cols.append(df.shift(i))
#         names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
#     # forecast sequence (t, t+1, ... t+n)
#     for i in range(0, n_out):
#         cols.append(df.shift(-i))
#         if i == 0:
#             names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
#         else:
#             names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
#     # put it all together
#     agg = pd.concat(cols, axis=1)
#     agg.columns = names
#     # drop rows with NaN values
#     if dropnan:
#         agg.dropna(inplace=True)
#     return agg
#
#
# def cs_to_sl():
#     # load dataset
#     dataset = pd.read_csv('pollution.csv', header=0, index_col=0)
#     values = dataset.values
#     # integer encode direction
#     encoder = LabelEncoder()
#     values[:, 4] = encoder.fit_transform(values[:, 4])
#     # ensure all data is float
#     values = values.astype('float32')
#     # normalize features
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled = scaler.fit_transform(values)
#     # frame as supervised learning
#     reframed = series_to_supervised(scaled, 1, 1)
#     # drop columns we don't want to predict
#     reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
#     print(reframed.head())
#     return reframed, scaler
#
#
# def train_test(reframed):
#     # split into train and test sets
#     values = reframed.values
#     n_train_hours = 365 * 24
#     train = values[:n_train_hours, :]
#     test = values[n_train_hours:, :]
#     # split into input and outputs
#     train_X, train_y = train[:, :-1], train[:, -1]
#     test_X, test_y = test[:, :-1], test[:, -1]
#     # reshape input to be 3D [samples, timesteps, features]
#     train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
#     test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
#     print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#     return train_X, train_y, test_X, test_y
#
#
# def fit_network(train_X, train_y, test_X, test_y, scaler):
#     model = Sequential()
#     model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
#     model.add(Dense(1))
#     model.compile(loss='mae', optimizer='adam')
#     # fit network
#     history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
#                         shuffle=False)
#     # plot history
#     pyplot.plot(history.history['loss'], label='train')
#     pyplot.plot(history.history['val_loss'], label='test')
#     pyplot.legend()
#     pyplot.show()
#     # make a prediction
#     yhat = model.predict(test_X)
#     test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
#     # invert scaling for forecast
#     inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
#     inv_yhat = scaler.inverse_transform(inv_yhat)
#     inv_yhat = inv_yhat[:, 0]
#     # invert scaling for actual
#     inv_y = scaler.inverse_transform(test_X)
#     inv_y = inv_y[:, 0]
#     # calculate RMSE
#     rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
#     print('Test RMSE: %.3f' % rmse)
#
#
# if __name__ == '__main__':
#     drow_pollution()
#     reframed, scaler = cs_to_sl()
#     train_X, train_y, test_X, test_y = train_test(reframed)
#     fit_network(train_X, train_y, test_X, test_y, scaler)
import tensorflow as tf
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print(tf.test.is_gpu_available())