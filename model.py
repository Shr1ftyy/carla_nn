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

# Network for semantic segmentation
class SegNet(layers.Layer):
    def __init__(self, in_shape, channel_idx="channels_last"):
        super(Conv, self).__init__()
        self.channel_idx = channel_idx
        self.in_shape = in_shape

        self.maxpool = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None, **kwargs)
        # self.Bn = layers.BatchNormalization()

        self.Conv1 = layers.Conv2D(64, 3, strides=1, padding='same', data_format=channel_idx, input_shape=self.in_shape, activation='elu')
        self.Conv2 = layers.Conv2D(128, 3, strides=1, padding='same', data_format=channel_idx, input_shape=self.in_shape, activation='elu')
        self.Conv3 = layers.Conv2D(256, 3, strides=1, padding='same', data_format=channel_idx, input_shape=self.in_shape, activation='elu')
        self.Conv4 = layers.Conv2D(512, 3, strides=1, padding='same', data_format=channel_idx, input_shape=self.in_shape, activation='elu')

        self.upsample = layers.UpSampling2D(size=(2,2), data_format=channel_idx)

        self.Deconv1 = layers.Conv2DTranspose(64, 3, strides=1, padding='same', data_format=channel_idx, input_shape=self.in_shape, activation='elu')
        self.Deconv2 = layers.Conv2DTranspose(128, 3, strides=1, padding='same', data_format=channel_idx, input_shape=self.in_shape, activation='elu')
        self.Deconv3 = layers.Conv2DTranspose(256, 3, strides=1, padding='same', data_format=channel_idx, input_shape=self.in_shape, activation='elu')
        self.Deconv4 = layers.Conv2DTranspose(512, 3, strides=1, padding='same', data_format=channel_idx, input_shape=self.in_shape, activation='elu')

    def call(self, inputs):
        # Encoding
        X = self.Conv1(inputs)
        X = self.Conv1(X)
        X = self.maxpool(X)

        X = self.Conv2(X)
        X = self.Conv2(X)
        X = self.maxpool(X)

        X = self.Conv3(X)
        X = self.Conv3(X)
        X = self.maxpool(X)

        X = self.Conv4(X)
        X = self.Conv4(X)
        X = self.maxpool(X)

        # Decoding
        X = self.upsample(X)
        X = self.Deconv1(X)
        X = self.Deconv1(X)

        X = self.upsample(X)
        X = self.Deconv2(X)
        X = self.Deconv2(X)

        X = self.upsample(X)
        X = self.Deconv3(X)
        X = self.Deconv3(X)

        X = self.upsample(X)
        X = self.Deconv4(X)
        output = self.Deconv4(X)

        return output


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

    # def call(self, inputs, training=False):
    def call(self, inputs): 
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
