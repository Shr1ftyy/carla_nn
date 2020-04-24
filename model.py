import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

class ConvNet(tf.keras.Model):
    def __init__(self, img_h, img_w):
        super(ConvNet, self).__init__()
        self.IMG_H = img_h
        self.IMG_W = img_w
        self.inConv = layers.Conv2D(64, 3, strides=1, padding='same', data_format="channels_last", input_shape=(self.IMG_H, self.IMG_W, 3), activation='relu')
        self.maxpool2d = layers.MaxPool2D((2,2), padding='same', data_format='channels_last')
        self.hConv2d = layers.Conv2D(64, 3, strides=1, padding='same', data_format="channels_last", activation='relu')
        self.flatten = layers.Flatten(data_format="channels_last")
        self.dropout = layers.Dropout(0.2)
        self.dense100 = layers.Dense(50)
        self.dense3 = layers.Dense(3, activation='linear')

    def call(self, inputs, training=False):
        X = self.inConv(inputs)
        X = self.maxpool2d(X)
        X = self.hConv2d(X)
        X = self.maxpool2d(X)
        X = self.flatten(X)
        X = self.dropout(X)
        X = self.dense100(X)
        X = self.dropout(X)
        output = self.dense3(X)
        
        return output

