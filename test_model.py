import os
import sys
import time
import numpy as np
import utils
import argparse
import cv2
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Reshape, CuDNNLSTM, Flatten, MaxPool2D, Dense, Conv2D, Dropout, Add
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt

# Imports model
from model import ResNet, SegNet

tf.executing_eagerly()
gpu_options = tf.GPUOptions(allow_growth=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75, allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

'''

This is is the code that loads the neural network,
trains and tests it

TODO:
    - Figure out how to make custom layers, take full advantage of tensorflow's
    or pytorch's API
    - Figure out how to make CNN accept multi-image input
    - Implement LSTM after flattening (cnnLSTM)
    img = np.split(cv2.imread(f"{IMG_DIR}/{name}", 1), num_images)[0]
    img = cv2.resize(img, (IMG_W, IMG_H))
    images.append(img)
    print(images[0].shape)

'''


parser = argparse.ArgumentParser(description='plays images from a selected TXT_DIR')
parser.add_argument('img_dir', metavar='-i', type=str, nargs='?', help='directory to get image data')
parser.add_argument('txt_dir', metavar='-t ', type=str, nargs='?', help='directory to get controls data')
parser.add_argument('test_img', metavar='-v', type=str, nargs='?', help='directory to get test image data')
parser.add_argument('test_txt', metavar='-f', type=str, nargs='?', help='directory to get test controls data')
args = parser.parse_args()

IMG_DIR = args.img_dir
TST_DIR = args.test_img
TST_TXT = args.test_txt
TXT_DIR = args.txt_dir
IMG_H = int(480/4)
IMG_W = int(640/4)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75, allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

EPOCHS = 50
BATCH_SIZE = 32
model_name = f'{time.time}_carNN.h5'
num_images = 4

#get training data images and controls
imageNames = [] 
images = []
testImages = []
testNames = []

imgFolder = utils.imgsort(os.listdir(IMG_DIR))
testFolder = utils.imgsort(os.listdir(TST_DIR))

controls = []
testControls = []

controlFile = str(open(TXT_DIR, 'r').read()).split('\n')
testFile = str(open(TST_TXT, 'r').read()).split('\n')

for name in imgFolder:
    imageNames.append(f"{name}.png")

for name in testFolder:
    testNames.append(f"{name}.png")

print(testNames)

print('Loading Training Images')
for name in imageNames: 
    # print(f"{IMG_DIR}/{name}")
    # print(cv2.imread(f"{IMG_DIR}/{name}", 0))
    # TESTING WITH ONE IMAGE FOR NOW
    img = np.split(cv2.imread(f"{IMG_DIR}/{name}", 1), num_images)[0]
    img = cv2.resize(img, (IMG_W, IMG_H))
    images.append(img)

print('Loading Testing Images')
for name in testNames: 
    # TESTING WITH ONE IMAGE FOR NOW
    # testImages.append(np.asarray(np.split(cv2.imread(f"{TST_DIR}/{name}", 1), num_images)[0]))
    tst = np.split(cv2.imread(f"{TST_DIR}/{name}", 1), num_images)[0]
    tst = cv2.resize(tst, (IMG_W, IMG_H))
    testImages.append(tst)

print('Loading Training Controls')

for line in controlFile:
   controls.append(line.split(','))

controls.pop()
controls = np.array(controls).astype('float32')


print('Loading Testing Controls')
for line in testFile:
    testControls.append(line.split(','))

testControls.pop()

testControls = np.array(testControls).astype('float32')

print(np.shape(images))
print(np.shape(testImages))
print(np.shape(controls))
print(np.shape(testControls))
print(controls[len(controls)-1])
print(testControls[len(testControls)-1])

# Preprocessing
sc = MinMaxScaler(feature_range=(0, 1))

# testControls = np.asarray(testControls).astype('float32')
# controls = np.asarray(controls).astype('float32')
x_train = np.asarray(images)/255.0
x_train = np.delete(x_train, len(x_train)-1, axis=0)
print(x_train.shape)
# sys.exit()

x_test = np.asarray(testImages)/255.0
x_test = np.delete(x_test, len(x_test)-1, axis=0)

y_train = sc.fit_transform(np.asarray(controls))

# Prepare model for training
model = SegNet((IMG_H, IMG_W))

model.compile(optimizer = 'adam', loss= 'mean_squared_error')

model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

if not os.path.exists('models'):
    os.mkdir('models')

model.save_weights('models/res.h5', save_format='h5')

# Predict on testing dataset
predictions = model.predict(x_test)
predictions = sc.inverse_transform(predictions)
print(predictions)

# Show results and compare
plt.plot(testControls, color = 'blue', label = f'Real steering')
plt.plot(predictions, color = 'red', label = f'Predicted steering')
plt.title(f"Steering Angle Prediction")
plt.xlabel('Frame')
plt.ylabel('Steering Angle')
plt.legend()

plt.show()
