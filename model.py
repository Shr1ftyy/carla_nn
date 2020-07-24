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

        self.inConv = layers.Conv2D(64, 3, strides=1, data_format=self.channel_idx, activation='linear')
        self.maxpool2d = layers.MaxPool2D((1,1), data_format=self.channel_idx)
        self.hConv2d = layers.Conv2D(64, 3, strides=1, data_format=self.channel_idx, activation='linear')
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
class TestNet(tf.keras.Model):
    def __init__(self, in_shape, channel_idx="channels_last"):
        super(TestNet, self).__init__()
        self.in_shape = in_shape

        self.maxpool = layers.MaxPool2D(pool_size=(2, 2), data_format=channel_idx)
        self.Dropout = layers.Dropout

        self.inConv = layers.Conv2D(16, 3, data_format=channel_idx,padding='same', activation='elu')
        self.Conv1 = layers.Conv2D(16, 3,  data_format=channel_idx,padding='same',activation='elu')
        self.Conv2 = layers.Conv2D(32, 3,  data_format=channel_idx, padding='same',activation='elu')
        self.Conv3 = layers.Conv2D(64, 3,  data_format=channel_idx, padding='same',activation='elu')
        self.Conv4 = layers.Conv2D(128, 3, data_format=channel_idx,padding='same',activation='elu')
        self.Conv5 = layers.Conv2D(256, 3, data_format=channel_idx,padding='same',activation='elu')
        self.concatenate = layers.concatenate

        self.upsample = layers.UpSampling2D(size=(2,2), data_format=channel_idx)

        self.Deconv4 = layers.Conv2DTranspose(16, 3,  data_format=channel_idx,padding='same',input_shape=self.in_shape, activation='elu')
        self.Deconv3 = layers.Conv2DTranspose(32, 3,  data_format=channel_idx,padding='same',input_shape=self.in_shape, activation='elu')
        self.Deconv2 = layers.Conv2DTranspose(64, 3,  data_format=channel_idx,padding='same',input_shape=self.in_shape, activation='elu')
        self.Deconv1 = layers.Conv2DTranspose(128, 3,data_format=channel_idx,padding='same',input_shape=self.in_shape, activation='elu')
                                                                                         
    def call(self, inputs):                                                              
        c1 = self.inConv(inputs)
        c1 = self.Dropout(0.1)(c1)
        c1 = self.Conv1(c1)
        p1 = self.maxpool(c1)

        c2 = self.Conv2(p1)
        c2 = self.Dropout(0.1)(c2)
        c2 = self.Conv2(c2)
        p2 = self.maxpool(c2)
         
        c3 = self.Conv3(p2)
        c3 = self.Dropout(0.2)(c3)
        c3 = self.Conv3(c3)
        p3 = self.maxpool(c3)
         
        c4 = self.Conv4(p3)
        c4 = self.Dropout(0.2)(c4)
        c4 = self.Conv4(c4)
        p4 = self.maxpool(c4)
         
        c5 = self.Conv5(p4)
        c5 = self.Dropout(0.3)(c5)
        c5 = self.Conv5(c5)

        u6 = self.Deconv1(c5)
        u6 = self.concatenate([u6, c4])
        c6 = self.Conv4(u6)
        c6 = self.Dropout(0.2)(c6)
        c6 = self.Conv4(c6)
         
        u7 = self.Deconv2(c6)
        u7 = self.concatenate([u7, c3])
        c7 = self.Conv3(u7)
        c7 = self.Dropout(0.2)(c7)
        c7 = self.Conv3(c7)
         
        u8 = self.Deconv3(c7)
        u8 = self.concatenate([u8, c2])
        c8 = self.Conv2(u8)
        c8 = self.Dropout(0.1)(c8)
        c8 = self.Conv2(c8)
         
        u9 = self.Deconv4(c8)
        u9 = self.concatenate([u9, c1], axis=3)
        c9 = self.Conv1(u9)
        c9 = self.Dropout(0.1)(c9)
        output = self.Conv1(c9)

        return output
        


class SegNet(tf.keras.Model):
    def __init__(self, in_shape, channel_idx="channels_last"):
        super(SegNet, self).__init__()
        self.channel_idx = channel_idx
        self.in_shape = in_shape

        self.maxpool = layers.MaxPool2D(pool_size=(2, 2), data_format=channel_idx)
        # self.Bn = layers.BatchNormalization()

        self.inConv = layers.Conv2D(64, 3, strides=1, data_format=channel_idx, input_shape=self.in_shape, activation='elu')
        self.Conv1 = layers.Conv2D(64, 3,  strides=1, data_format=channel_idx, activation='elu')
        self.Conv2 = layers.Conv2D(128, 3, strides=1, data_format=channel_idx,activation='elu')
        self.Conv3 = layers.Conv2D(256, 3, strides=1, data_format=channel_idx,activation='elu')
        self.Conv4 = layers.Conv2D(512, 3, strides=1, data_format=channel_idx,activation='elu')

        self.upsample = layers.UpSampling2D(size=(2,2), data_format=channel_idx)

        self.Deconv1 = layers.Conv2DTranspose(64, 3, strides=1, data_format=channel_idx, input_shape=self.in_shape, activation='elu')
        self.Deconv2 = layers.Conv2DTranspose(128, 3, strides=1, data_format=channel_idx, input_shape=self.in_shape, activation='elu')
        self.Deconv3 = layers.Conv2DTranspose(256, 3, strides=1, data_format=channel_idx, input_shape=self.in_shape, activation='elu')
        self.Deconv4 = layers.Conv2DTranspose(512, 3, strides=1, data_format=channel_idx, input_shape=self.in_shape, activation='elu')

    def call(self, inputs):
        # Encoding
        X = self.inConv(inputs)
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
        X = self.Deconv4(X)

        return output


# CNN with a Residual Layer
class ResNet(tf.keras.Model):
    def __init__(self, in_shape, channel_idx="channels_last"):
        super(ResNet, self).__init__()
        self.channel_idx = channel_idx
        self.in_shape = in_shape

        self.inConv = layers.Conv2D(64, 3, strides=1, data_format=channel_idx, input_shape=self.in_shape, activation='linear')
        self.maxpool2d = layers.MaxPool2D((2,2), data_format=self.channel_idx)
        self.hConv2d = layers.Conv2D(64, 3, strides=1, data_format=self.channel_idx, activation='linear')
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
