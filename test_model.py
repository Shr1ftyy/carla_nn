import os
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
from model import ConvNet

gpu_options = tf.GPUOptions(allow_growth=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75, allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

'''

This is is the code that create the neural network,
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
    - Deploy the model train in realtime

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
IMG_H = 480/8
IMG_W = 640/8

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75, allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

EPOCHS = 5
BATCH_SIZE = 16
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

c = 0
for name in imgFolder:
    imageNames.append(f"{name}.png")

for name in testFolder:
    testNames.append(f"{name}.png")

print(imageNames)

for name in imageNames: 
    print(f"{IMG_DIR}/{name}")
    print(cv2.imread(f"{IMG_DIR}/{name}", 0))
    # TESTING WITH ONE IMAGE FOR NOW
    img = np.split(cv2.imread(f"{IMG_DIR}/{name}", 1), num_images)[0]
    img = cv2.resize(img, (IMG_W, IMG_H))
    images.append(img)
    print(images[0].shape)
    c += 1
    if c >= 10:
        break

c = 0
for name in testNames: 
    # TESTING WITH ONE IMAGE FOR NOW
    # testImages.append(np.asarray(np.split(cv2.imread(f"{TST_DIR}/{name}", 1), num_images)[0]))
    tst = np.split(cv2.imread(f"{TST_DIR}/{name}", 1), num_images)[0]
    tst = cv2.resize(tst, (IMG_W, IMG_H))
    testImages.append(tst)
    print(images[0].shape)
    break

for line in controlFile:
    controls.append(line.split(','))
    c += 1
    if c >= 11:
        break

controls.pop()

for line in testFile:
    testControls.append(line.split(','))
    break

testControls.pop()

controls = np.array(controls)
testControls = np.array(testControls)

print(np.shape(images[0]))
print(np.shape(controls))
print(controls)

x_train = np.asarray(images)/255.0
x_test = np.asarray(testImages)/255.0
y_train = np.asarray(controls)


print(np.shape(x_test))

# Prepare model for training
# model = ConvNet(IMG_H, IMG_W)

model = Sequential()

model.add(Conv2D(64, 3, strides=1, padding='same', data_format="channels_last", input_shape=(IMG_H, IMG_W, 3), activation='relu'))
model.add(MaxPool2D((2,2), padding='same', data_format='channels_last'))
model.add(Conv2D(64, 3, strides=1, padding='same', data_format="channels_last", activation='relu'))
model.add(MaxPool2D((2,2), padding='same', data_format='channels_last'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dropout(0.2))
model.add(Dense(3))

model.compile(optimizer = 'adam', loss= 'mean_squared_error')
# model.save('test.h5')

model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

predictions = model.predict(x_test)

plt.plot(testControls[1], color = 'red', label = f'Real steering')
plt.plot(predictions[1], color = 'orange', label = f'Predicted steering')
plt.plot(testControls[0], color = 'blue', label = f'Real steering')
plt.plot(predictions[0], color = 'purple', label = f'Predicted steering')
plt.title(f"Steering Angle Prediction")
plt.xlabel('Frame')
plt.ylabel('Steering Angle')
plt.legend()

plt.show()
