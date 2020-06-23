#!C:/Users/Syeam/AppData/Local/Programs/Python/Python37/python.exe
# @author Syeam_Bin_Abdullah ------->  A script for playing around with Carla's API :D
import glob
import math
import cv2
import os
import sys
import numpy as np

try:
    sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major, 
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("Failed to find carla's .egg directory")

import carla
from carla import *
from car_env import CarEnv
import random
import time

run = False 
PORT = 2069
IMG_WIDTH = 640
IMG_HEIGHT = 480
spawn = CarEnv(port=2069) # <-- Remove when not testing :D 
spawn.vehicle_list[0].set_autopilot(enabled=True)

#black = np.zeros(shape=(IMG_HEIGHT, IMG_WIDTH,3))
CAMERA_MEM = None# --> stores current frame received from cameras - inits as empty (None)

def processImage(data, folder):
    global CAMERA_MEM
    i = np.array(data.raw_data)
    i2 = np.reshape(i, (IMG_HEIGHT, IMG_WIDTH, 4))
    i3 = i2[:, :, :3]
    CAMERA_MEM = i3
    data.save_to_disk(f'{folder}/%06d.png' % data.frame)
    print(dir(data))
    print(type(CAMERA_MEM))
    # return i3/255.0


def showCameras():
    # for i in range(0, len(CAMERA_MEM)):
        # print(f"{sensors}\n{len(sensors)}")
        try:
            cv2.imshow(f'camera', CAMERA_MEM)
            cv2.waitKey(50)
        except:
            pass    


client = carla.Client('localhost', PORT)
client.set_timeout(10)
world = client.get_world()
actor_list = world.get_actors()

foundCam = False

camNum = 0
for actor in spawn.sensor_list:
    if actor.attributes['role_name'] == 'frontCam':
        # CAMERA_MEM.append(None)
        foundCam = True
        actor.listen(lambda image: processImage(image, 'raw'))
        camNum+=1

    if actor.attributes['role_name'] == 'semantic':
        # CAMERA_MEM.append(None)
        foundCam = True
        actor.listen(lambda image: processImage(image, 'semantic'))
        camNum+=1

if foundCam:
    print('Found some Cameras!')
else:
    print('Failed to find cameras... Exiting')
    exit()
    sys.exit()

print("SENSOR INIT DONE")                       

run = True
while run == True:
    try:
    	# v = car.get_velocity()
	 # kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
         # try:
         	# os.system('cls')
         # except:
         	# os.system('clear')

         # print(f"""
         	# >>>Vehicle Status<<<
         	# Velocity: {kmh}kmh
         	# Location: {car.get_location()}
         	# """)

        showCameras()

    except (KeyboardInterrupt, SystemExit):
        spawn.destroy()
        sys.exit()
        exit()


##### NEED TO FIGURE OUT HOW TO FORMAT RADAR INPUT AND VISUALIZE :D


