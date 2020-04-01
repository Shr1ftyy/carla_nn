# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# @author Syeam_Bin_Abdullah ------->  A script for playing around with Carla's API :D
import glob
import math
import cv2
import os
import sys
import numpy as np

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

run = False 
PORT = 2069
IMG_WIDTH = 640
IMG_HEIGHT = 480

#black = np.zeros(shape=(IMG_HEIGHT, IMG_WIDTH,3))
CAMERA_MEM = np.array([None,None,None,None]) # --> stores current frame received from cameras - inits as empty (None)
RADAR_MEM = [None] # --> stores points in cloud from radar input
LIDAR_MEM = [None]
FPS = 30 # --> sets maximum FPS for sensor inputs
TICKRATE = 1/FPS # --> sets sensor tickrate 

def processImage(data, sensorID):
	i = np.array(data.raw_data)
	i2 = np.reshape(i, (IMG_HEIGHT, IMG_WIDTH, 4))
	i3 = i2[:, :, :3]
	CAMERA_MEM[sensorID] = i3
	# return i3/255.0


def showCameras():
	for i in range(0, len(CAMERA_MEM)):
		# print(f"{sensors}\n{len(sensors)}")
		try:
			cv2.imshow(f'camera_{i}', CAMERA_MEM[i])
			cv2.waitKey(50)
		except:
			pass    


client = carla.Client('localhost', PORT)
client.set_timeout(10)
world = client.get_world()
actor_list = world.get_actors()

for actor in actor_list:
     if actor.type  == 'sensor.camera.rgb': 
        if not actor.is_listening:
            actor.listen(lambda raw: processImage(raw))


         frontCam = actor
         print('found front camera!')
     else:
         print('could not find front camera!')
         exit()

exit()


print("SENSOR INIT DONE")                       



run = True
while run == True:
#	v = car.get_velocity()
#	kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
	# try:
	# 	os.system('cls')
	# except:
	# 	os.system('clear')

	# print(f"""
	# 	>>>Vehicle Status<<<
	# 	Velocity: {kmh}kmh
	# 	Location: {car.get_location()}
	# 	""")

	time.sleep(0.2)
	showCameras()



##### NEED TO FIGURE OUT HOW TO FORMAT RADAR INPUT AND VISUALIZE :D

