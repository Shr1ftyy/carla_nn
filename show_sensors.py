# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# @author Syeam_Bin_Abdullah ------->  A script for playing around with Carla's API :D
import glob
import math
import cv2
import os
import sys
import numpy as np

from car_env import CarEnv

# from matplotlib import pyplot as plt

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
#import pygame
#pygame.init()

#l_windim = (480, 640)

#win = pygame.display.set_mode(l_windim) # Window for displaying Lidar input



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

def processRadar(data):
	# To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
	raw_data = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
	time.sleep(0.5)
	points = np.reshape(raw_data, (int(len(raw_data)/4), 4))
	print(f'{points}')

	#   for detect in data:
	#       detect.azimuth = math.degrees(detect.azimuth)
	#       detect.altitude = math.degrees(detect.altitude)
	RADAR_MEM[0] = points


def processLidar(data): # Just experimenting with lidar, although not planning to be used in final model of car
	# points = np.array(data.raw_data)
	# points_len = int(len(points)/3)
	# points_reshaped = np.reshape(points, (points_len, 3))
	points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
	points = np.reshape(points, (int(points.shape[0] / 3), 3))  
	lidar_data = np.array(points[:, :2])
	lidar_data *= min(l_windim) / 100.0
	lidar_data += (0.5 * l_windim[1], 0.5 * l_windim[0])
	lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
	lidar_data = lidar_data.astype(np.int32)
	lidar_data = np.reshape(lidar_data, (-1, 2))
	lidar_img_size = (self.hud.dim[1], self.hud.dim[0], 3)
	lidar_img = np.zeros((lidar_img_size), dtype = int)
	lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
	self.surface = pygame.surfarray.make_surface(lidar_img)

#    LIDAR_MEM[0]= points_reshaped

def showCameras():
	for i in range(0, len(CAMERA_MEM)):
		# print(f"{sensors}\n{len(sensors)}")
		try:
			cv2.imshow(f'camera_{i}', CAMERA_MEM[i])
			cv2.waitKey(50)
		except:
			pass    

def showLidar():
	try:
		for point in LIDAR_MEM[0]:
			pygame.draw.rect(win, white, (point[0], point[1], 1, 1))
			pygame.display.update()        
	except:
		pygame.display.update()
#
#def showRadar():
#	carRect = (l_windim[0]/2, l_windim[1]/2, l_windim[0]/10, l_windim[0]/10)
#	pygame.draw.rect(win, (255,0,0), carRect)
#	front = (carRect[0], carRect[1])
#	try:
#		for point in RADAR_MEM[0]:
#			depth = point[3]
#			azimuth = math.degrees(point[2])
#			pygame.draw.rect(win, (255,255,255), (front[0]+int(depth*math.cos(azimuth)) if azimuth>=0 else front[0]-int(depth*math.cos(azimuth)), front[1]+int(depth*math.sin(azimuth)) if azimuth>=0 else front[1]-int(depth*math.sin(azimuth))), 5, 5) # TODO: Figure out x-axis positioning from angle and y offset 
#			pygame.display.update()
#	except:
#		pygame.display.update()		


car_env = CarEnv()
car = car_env.vehicle_list[0]
sensors = car_env.sensor_list

#listen for sensor input
# sensors[0].listen(lambda data: processRadar(data)) >>>> NEED TO FIGURE OUT RADAR VISUALIZATION IN THE FUTURE D:
sensors[1].listen(lambda image: processImage(image, 0))
sensors[2].listen(lambda image: processImage(image, 1))
sensors[3].listen(lambda image: processImage(image, 2))
sensors[4].listen(lambda image: processImage(image, 3))



print("SENSOR INIT DONE")                       


run = True
while run == True:
	v = car.get_velocity()
	kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
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
	win.fill((0,0,0))
	pygame.display.update()



##### NEED TO FIGURE OUT HOW TO FORMAT RADAR INPUT AND VISUALIZE :D
