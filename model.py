import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

# Residual Layer
class ResBlock Layer(layers.Layer):
    def __init__(self, channel_idx="channels_last", input_shape):
        super(ResBlock, self).__init__()
        self.channel_idx = channel_idx
        self.input_shape = input_shape
        self.inConv = layers.Conv2D(64, 3, strides=1, padding='same', data_format=self.channel_idx, input_shape=input_shape, activation='linear')
        self.maxpool2d = layers.MaxPool2D((2,2), padding='same', data_format=self.channel_idx)
        self.hConv2d = layers.Conv2D(64, 3, strides=1, padding='same', data_format=self.channel_idx, activation='linear')
        self.BatchNorm = layers.BatchNormalization()
        self.relu = Activation('relu')
        self.Add = Add()

    def call(self, inputs, training=False):
        X = self.inConv(inputs)
        X = self.BatchNorm(X)
        X = self.Conv2D(X)
        X = self.Add([X, inputs])
        X = self.relu(X)

        return X

# CNN with a Residual Layer
class ResNet(tf.keras.Model):
    def __init__(self, channel_idx="channels_last", input_shape, channels=None):
        super(ConvNet, self).__init__()
        self.channel_idx = channel_idx
        self.input_shape = input_shape
        self.inConv = layers.Conv2D(64, 3, strides=1, padding='same', data_format=channel_idx, input_shape=self.input_shape, activation='linear')
        self.maxpool2d = layers.MaxPool2D((2,2), padding='same', data_format=self.channel_idx)
        self.hConv2d = layers.Conv2D(64, 3, strides=1, padding='same', data_format=self.channel_idx, activation='linear')
        self.flatten = layers.Flatten(data_format=self.channel_idx)
        self.dropout = layers.Dropout(0.2)
        self.dense100 = layers.Dense(100)
        self.dense3 = layers.Dense(3, activation='linear')
        self.ResBlock = ResBlock(channel_idx=self.channel_idx)

    def call(self, inputs, training=False):
        X = self.inConv(inputs)
        X = self.hConv2d(X)
        X = self.maxpool2d(X)
        X = self.ResBlock(X)
        X = self.hConv2d(X)
        X = self.flatten(X)
        X = self.dropout(X)
        X = self.dense100(X)
        X = self.dropout(X)
        output = self.dense3(X)
        
        return output

