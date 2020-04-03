from car_env import CarEnv
import sys
import glob 
import numpy as np 
import os
import cv2
import argparse
#import pygame

try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
                    sys.version_info.major, 
                    sys.version_info.minor,
                    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("Failed to find carla's .egg TXT_DIR")

import carla
from carla import *

import random
import time

parser = argparse.ArgumentParser(description='plays images from a selected TXT_DIR')
parser.add_argument('img_dir', metavar='img_dir', type=str, nargs='?', help='directory to store image data')
parser.add_argument('txt_dir', metavar='txt_dir', type=str, nargs='?', help='directory to store controls data')
parser.add_argument('txt_name', metavar='txt_name', type=str, nargs='?', help='name of text file to store controls data')
args = parser.parse_args()

IMG_DIR = args.img_dir
TXT_DIR = args.txt_dir
TXT_NAME = args.txt_name

car_env = CarEnv(port=2069)
car = car_env.vehicle_list[0]
sensors = car_env.sensor_list
im_width = car_env.im_width
im_height = car_env.im_height
CAMERA_MEM = []
timestamp = 0


# Prints controls into console
def showLogs():
    controls = car.get_control()   
    print(f"""Throttle:{controls.throttle},Steering:{controls.steer}, Brake:{controls.brake}""")

# Places image into memory (list)
def processimg(data, sid):
    i = np.array(data.raw_data)
    i2 = np.reshape(i, (im_height, im_width, 4))
    i3 = i2[:, :, :3]
    CAMERA_MEM[sid] = i3


# Gathers steering, brake, throttle, current camera feed, and saves
def gather_data():
    global timestamp
    controls = car.get_control()
    throttle = controls.throttle
    brake  = controls.brake
    steer = controls.steer

    for j in range(0, len(sensors)):
        os.chdir(IMG_DIR)
        stitch = np.concatenate(CAMERA_MEM[:])
        cv2.imwrite(f'{timestamp}.png', stitch)
        os.chdir('../..')

    timestamp += 1

    return f"{throttle},{steer},{brake}"


# Main function
def main():
    global data
    global sensors
    print('initializing memory')
    for _ in sensors:
        CAMERA_MEM.append(None)

    sensors[0].listen(lambda image: processimg(image, 0))
    sensors[1].listen(lambda image: processimg(image, 1))
    sensors[2].listen(lambda image: processimg(image, 2))
    sensors[3].listen(lambda image: processimg(image, 3))

    if not os.path.exists():
        os.mkdir(TXT_DIR)
        print('made dir')

    if not os.path.exists(IMG_DIR):
        os.mkdir(IMG_DIR)
        print('made img dir')

    data = open(f"{TXT_DIR}{TXT_NAME}", "w")
    print('initializing sensors')
    time.sleep(5)

    #Activates autopilot -> Data is collected while autopilot is running 
    car.set_autopilot(enabled=True)

    try:
        while timestamp < 2000:
            showLogs()
            data.write(f'{gather_data()}\n')
        data.close()
        car.destroy()
        for sensor in sensors:
            sensor.destroy()
        sys.exit()
        exit()
        raise
    except (KeyboardInterrupt, SystemExit):
        data.close()
        car.destroy()
        for sensor in sensors:
            sensor.destroy()
        sys.exit()
        exit()
        raise

if __name__ == '__main__':
    main()
else:
    print("This module is cannot be imported -> quitting now")
    exit()
