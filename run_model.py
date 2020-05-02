import os
from model import ConvNet
from car_env import CarEnv
import sys
import glob
import time
import numpy as np
import utils
import argparse
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import BatchNormalization, Reshape, CuDNNLSTM, Flatten, MaxPool2D, Dense, Conv2D, Dropout, Add

try:
    sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major, 
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("Failed to find carla's .egg directory")
import carla
from carla import *

import random

parser = argparse.ArgumentParser(description='loads a model and runs it')
parser.add_argument('model', metavar='-m', type=str, nargs='?', help='directory to get model')
args = parser.parse_args()

# Some variables
IMAGE_MEM = []

car_env = CarEnv(port=2069)
car = car_env.vehicle
sensors = car_env.sensor_list

SCL = 4
img_h = int(car_env.im_height/SCL)
img_w = int(car_env.im_width/SCL)


def clean():
    car_env.destroy()
    sys.exit()

def processimg(data, sensorID):
    i = np.array(data.raw_data)
    i2 = np.reshape(i, (img_h*SCL, img_w*SCL, 4))
    i3 = i2[:, :, :3]
    i3 = cv2.resize(i3/255.0, (img_w, img_h))
    i3 = np.expand_dims(i3, axis=0)
    IMAGE_MEM[sensorID] = i3

print(img_w)

# Loads model weights
# model = ConvNet(img_h=img_h,img_w=img_w)

model = Sequential()

model.add(Conv2D(64, 3, strides=1, padding='same', data_format="channels_last", input_shape=(img_h, img_w, 3), activation='linear'))
model.add(MaxPool2D((2,2), padding='same', data_format='channels_last'))
model.add(Conv2D(64, 3, strides=1, padding='same', data_format="channels_last", activation='linear'))
model.add(MaxPool2D((2,2), padding='same', data_format='channels_last'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Dense(3, activation='linear'))

model.load_weights(args.model)

foundCam = False
for sensor in sensors:
    if sensor.attributes['role_name']  == 'frontCam': 
        IMAGE_MEM.append(None)
        print('Found front camera')
        sensor.listen(lambda image: processimg(image, 0))
        foundCam = True

if not foundCam:
    print('Could not find any cameras, exiting...')
    clean()

time.sleep(5)

while True:
    try:
        pred = model.predict(IMAGE_MEM[0]) 
        pred = pred[0]
        print(pred)
        car.apply_control(carla.VehicleControl(throttle=float(pred[0]), steer=float(pred[1])))
        try:
            cv2.imshow('Preview', IMAGE_MEM[0][0])
            cv2.waitKey(1)
        except:
            pass
    except (KeyboardInterrupt, SystemExit):
        clean()
