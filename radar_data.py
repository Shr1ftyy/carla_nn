from car_env import CarEnv
import sys
import glob 
import numpy as np 
import os
import cv2
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

RADAR_MEM = []

def save_data(radar_data, sid):
    # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
    points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (len(radar_data), 4))
    RADAR_MEM.append(points)

def main():
    global RADAR_MEM
    data = open('radar.txt', 'w')
    try:
        car_env = CarEnv(port=2069)
        sensors = car_env.sensor_list
        car = car_env.vehicle_list[0]
        car.set_autopilot(enabled=True)

        for sensor in sensors:
            if sensor.type_id == 'sensor.other.radar':
                sensor.listen(lambda data: save_data(data, 0))

        time.sleep(5)

        while True:
            for point in RADAR_MEM:
                if len(RADAR_MEM) % 50 == 0: 
                    RADAR_MEM = []
                    data.write('\n')
                if point.size != 0:
                    stuff = str(str(point).replace('[', '')).replace(']', '')
                    print(stuff)
                    data.write(f"{stuff}|")
                else:
                    pass


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
    print('you cannot import this module - radar_data')
