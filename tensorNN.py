import os
import time
import numpy as np
import utils
import argparse
import cv2
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Reshape, CuDNNLSTM, Flatten, MaxPool2D, Dense, Conv2D, Dropout
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt

'''

This is is the code that create the neural network,
trains and tests it

TODO:
    - Figure out how to make custom layers, take full advantage of tensorflow's
    or pytorch's API
    - Figure out how to make CNN accept multi-image input
    - Implement LSTM after flattening (cnnLSTM)
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
IMG_H = 480
IMG_W = 640

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75, allow_growth=True)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

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

for name in imgFolder:
    imageNames.append(f"{name}.png")

for name in testFolder:
    testNames.append(f"{name}.png")

print(imageNames)

for name in imageNames: 
    print(f"{IMG_DIR}/{name}")
    print(cv2.imread(f"{IMG_DIR}/{name}", 0))
    # TESTING WITH ONE IMAGE FOR NOW
    images.append(np.split(cv2.imread(f"{IMG_DIR}/{name}", 1), num_images)[0])
    print(images[0].shape)

for name in testNames: 
    # TESTING WITH ONE IMAGE FOR NOW
    testImages.append(np.split(cv2.imread(f"{TST_DIR}/{name}", 1), num_images)[0])

for line in controlFile:
    controls.append(line.split(','))
controls.pop()

for line in testFile:
    testControls.append(line.split(','))
testControls.pop()

controls = np.array(controls)
testControls = np.array(testControls)

print(np.shape(images[0]))
print(np.shape(controls))
print(controls)

x_train = np.asarray(images)
x_test = np.asarray(testImages)
y_train = np.asarray(controls)

print(np.shape(x_test))

# attempting to customize a layer :\ 
def customFlatten(layer, batch_size, seq_len):
    pass

# Model
model = Sequential([
    # Conv2D(64, 3, strides=1, padding='same', data_format="channels_first", input_shape=(4, IMG_H, IMG_W), activation='relu'), 
    Conv2D(64, 3, strides=1, padding='same', data_format="channels_last", input_shape=(IMG_H, IMG_W, 3), activation='relu'), 
    BatchNormalization(axis=1),
    MaxPool2D((2,2), padding='same', data_format='channels_last'),
    Conv2D(64, 3, strides=1, padding='same', data_format="channels_last", activation='relu'),
    BatchNormalization(axis=1),
    MaxPool2D((2,2), padding='same', data_format='channels_last'),
    Flatten(data_format="channels_last"),
    # CuDNNLSTM(units=64, return_sequences=True, input_shape=(None, )),
    Dropout(0.2),
    Dense(100),
    Dense(3),
])

model.compile(optimizer = 'adam', loss= 'mean_squared_error')
model.save('test.h5')


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
