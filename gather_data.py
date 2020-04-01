from car_env import CarEnv
import sys
import glob 
import numpy as np 
import math
import cv2
import os
#import pygame

try:
		sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
				sys.version_info.major, 
				sys.version_info.minor,
				'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
		print("Failed to find carla's .egg directory")

import carla
from carla import *

import random
import time

car_env = CarEnv(port=2069)
car = car_env.vehicle_list[0]
sensors = car_env.sensor_list
im_width = car_env.im_width
im_height = car_env.im_height
CAMERA_MEM = [None]
timestamp = 0
DIRECTORY = 'data\\'
IMG_DIR = f"images/{timestamp}.png"

#Activate autopilot (built-in carla function)
car.set_autopilot(enabled=True)

#Functions for gathering data from the car
#Get controls and imagery via threading

def processImage(data, sensorID):
	i = np.array(data.raw_data)
	i2 = np.reshape(i, (im_height, im_width, 4))
	i3 = i2[:, :, :3]
	CAMERA_MEM[sensorID] = i3
	timestamp += 1 


def gather_data():
    controls = car.get_control()
    throttle = controls.throttle
    brake  = controls.brake
    steer = controls.steer
    try:
	    cv2.imwrite(IMG_DIR, CAMERA_MEM[0])
    except:
    	pass
   # v = car.get_velocity()
   # kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

    return f"{throttle},{steer},{brake}"

# Get controls applied by autopilot


def main():
    sensors[0].listen(lambda image: processImage(image, 0))

    if not os.path.exists(DIRECTORY):
        os.mkdir(DIRECTORY)
        print('made dir')

    data = open(f"{DIRECTORY}controls.txt", "w")
    print('initializing sensors')
    time.sleep(5)

    try:
        while True:
            try:
                os.system('cls')
            except:
                os.system('clear')

            controls = car.get_control()

            print(controls)
            print(f"""
                    Throttle: {controls.throttle}
                    Steering: {controls.steer}
                    Brake: {controls.brake}

                    """)
            data.write(f'{gather_data()}\n')
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
