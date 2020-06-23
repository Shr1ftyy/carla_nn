import tensorflow as tf
import tensorflow.keras.layers as layers
# from tensorflow.compat.v1.keras.layers import CuDDNLSTM
import numpy as np
tf.executing_eagerly()


# Residual Layer
class ResBlock(layers.Layer):
    def __init__(self, channel_idx="channels_last"):
        super(ResBlock, self).__init__()
        self.channel_idx = channel_idx
        # self.in_shape = in_shape

        self.inConv = layers.Conv2D(64, 3, strides=1, padding='same', data_format=self.channel_idx, activation='linear')
        self.maxpool2d = layers.MaxPool2D((1,1), padding='same', data_format=self.channel_idx)
        self.hConv2d = layers.Conv2D(64, 3, strides=1, padding='same', data_format=self.channel_idx, activation='linear')
        self.BatchNorm = layers.BatchNormalization()
        self.relu = layers.Activation('linear')
        self.Add = layers.Add()

    def call(self, inputs, training=False):
        X = self.inConv(inputs)
        X = self.BatchNorm(X)
        X = self.hConv2d(X)
        X = self.Add([X, inputs])
        X = self.relu(X)

        return X

# CNN with a Residual Layer
class ResNet(tf.keras.Model):
    def __init__(self, in_shape, channel_idx="channels_last"):
        super(ResNet, self).__init__()
        self.channel_idx = channel_idx
        self.in_shape = in_shape

        self.inConv = layers.Conv2D(64, 3, strides=1, padding='same', data_format=channel_idx, input_shape=self.in_shape, activation='linear')
        self.maxpool2d = layers.MaxPool2D((2,2), padding='same', data_format=self.channel_idx)
        self.hConv2d = layers.Conv2D(64, 3, strides=1, padding='same', data_format=self.channel_idx, activation='linear')
        self.flatten = layers.Flatten(data_format=self.channel_idx)
        self.dropout = layers.Dropout(0.2)
        self.dense1000 = layers.Dense(1000)
        self.dense100 = layers.Dense(100)
        self.dense1 = layers.Dense(1, activation='linear')
        self.ResBlock = ResBlock()
        # self.lstm = layer.LSTM(units=500, return_sequences=True, return_state=True)

    def call(self, inputs, training=False):
        X = self.inConv(inputs)
        X = self.hConv2d(X)
        X = self.ResBlock(X)
        X = self.ResBlock(X)
        X = self.ResBlock(X)
        X = self.flatten(X)
        # X = self.lstm(X)
        X = self.dropout(X)
        X = self.dense100(X)
        X = self.dropout(X)
        X = self.dense100(X)
        X = self.dropout(X)
        output = self.dense1(X)
        
        return output
