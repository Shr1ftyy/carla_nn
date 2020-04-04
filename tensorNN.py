import os
import time
import numpy as np
import utils
import argparse
import cv2
#import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Conv2D, Dropout
from tensorflow.keras.models import Sequential, load_model
#from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

parser = argparse.ArgumentParser(description='plays images from a selected TXT_DIR')
parser.add_argument('img_dir', metavar='img_dir', type=str, nargs='?', help='directory to get image data')
parser.add_argument('txt_dir', metavar='txt_dir', type=str, nargs='?', help='directory to get controls data')
parser.add_argument('test_img', metavar='test_img', type=str, nargs='?', help='directory to get test image data')
parser.add_argument('test_txt', metavar='test_txt', type=str, nargs='?', help='directory to get test controls data')
args = parser.parse_args()

IMG_DIR = args.img_dir
TST_DIR = args.test_dir
TST_TXT = args.test_txt
TXT_DIR = args.txt_dir

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

for name in imageNames: 
    images.append(np.split(cv2.imread(f"{IMG_DIR}{name}", 0), num_images))

for name in testNames: 
    images.append(np.split(cv2.imread(f"{TST_DIR}{name}", 0), num_images))

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

x_train = images
x_test = testImages
y_train = controls

print(np.shape(x_test))
exit()

model = Sequential()
model.add(Conv2D(512, 3, strides=1, padding='same', input_shape=(4, 480, 640)))
model.add(Reshape((512, -1)))
model.add(Permute((2, 1)))
model.add(CuDDNLSTM(32))
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Dense(3))

model.compile(optimizer = 'adam', loss= 'mean_squared_error')


model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

predictions = model.predict(x_test)


