# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# @author Syeam_Bin_Abdullah ------->  A script for playing around with Carla's API :D
import glob
import math
import cv2
import os
from tqdm import tqdm
import sys
import numpy as np
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
import pygame
#pygame.init()

l_windim = (480, 640)

win = pygame.display.set_mode(l_windim) # Window for displaying Lidar input



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


def processLidar(data):
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

def showRadar():
	carRect = (l_windim[0]/2, l_windim[1]/2, l_windim[0]/10, l_windim[0]/10)
	pygame.draw.rect(win, (255,0,0), carRect)
	front = (carRect[0], carRect[1])
	try:
		for point in RADAR_MEM[0]:
			depth = point[3]
			azimuth = math.degrees(point[2])
			pygame.draw.rect(win, (255,255,255), (front[0]+int(depth*math.cos(azimuth)) if azimuth>=0 else front[0]-int(depth*math.cos(azimuth)), front[1]+int(depth*math.sin(azimuth)) if azimuth>=0 else front[1]-int(depth*math.sin(azimuth))), 5, 5) # TODO: Figure out x-axis positioning from angle and y offset 
			pygame.display.update()
	except:
		pygame.display.update()		

class CarEnv:
	def __init__(self):
		self.im_width = IMG_WIDTH
		self.im_height = IMG_HEIGHT
		self.vehicle_list = [] # --> Vehicle List
		self.sensor_list = [] # --> Sensor List
		self.client = carla.Client('localhost', PORT)
		self.client.set_timeout(10)
		self.world = self.client.get_world()
		# self.settings = self.world.get_settings()
		# self.settings.no_rendering_mode = True
		# self.world.apply_settings(self.settings)
		self.blueprint_library = self.world.get_blueprint_library()
		# Find specific blueprint.
		spawn_points = self.world.get_map().get_spawn_points()
		self.spawn_point = random.choice(spawn_points)

		self.vehicle_bp = self.blueprint_library.filter("vehicle.tesla.model3")[0]
		self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.spawn_point)
		self.vehicle.set_autopilot(enabled=True)
		self.vehicle_list.append(self.vehicle)

		#SENSOR INIT
		self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
		self.rgb_cam.set_attribute("sensor_tick", f"{TICKRATE}")
		self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
		self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
		self.rgb_cam.set_attribute("fov", f"110")
		
		self.radar = self.blueprint_library.find('sensor.other.radar')
		self.radar.set_attribute("sensor_tick", f'{TICKRATE}')
		self.radar.set_attribute("horizontal_fov", f'110')
		self.radar.set_attribute("vertical_fov", f'45')
		self.radar.set_attribute("points_per_second", f'50')
		self.radar.set_attribute("range", f'20')

		self.frontTrans = carla.Transform(carla.Location(x=2.5, z=0.75))
		self.leftTrans = carla.Transform(carla.Location(x=2.5,y=-1, z=0.75), carla.Rotation(yaw=-90)) 
		self.rightTrans = carla.Transform(carla.Location(x=2.5, y=1, z=0.75), carla.Rotation(yaw=90))
		self.backTrans = carla.Transform(carla.Location(x=-2.5, z=0.75), carla.Rotation(yaw=180))

		# RADAR
		self.frontRadar = self.world.spawn_actor(self.radar, self.frontTrans, attach_to=self.vehicle_list[0])
		self.sensor_list.append(self.frontRadar)

		#DEPTH

		#BGRA CAMERAS
		self.frontCam = self.world.spawn_actor(self.rgb_cam, self.frontTrans, attach_to=self.vehicle_list[0])
		self.sensor_list.append(self.frontCam)
		self.leftCam = self.world.spawn_actor(self.rgb_cam, self.leftTrans, attach_to=self.vehicle_list[0])
		self.sensor_list.append(self.leftCam)
		self.rightCam= self.world.spawn_actor(self.rgb_cam, self.rightTrans, attach_to=self.vehicle_list[0])
		self.sensor_list.append(self.rightCam)
		self.backCam= self.world.spawn_actor(self.rgb_cam, self.backTrans, attach_to=self.vehicle_list[0])
		self.sensor_list.append(self.backCam)



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
	print(f'{RADAR_MEM[0]}')
	showRadar()
	# showLidar()
	win.fill((0,0,0))
	pygame.display.update()



##### NEED TO FIGURE OUT HOW TO FORMAT RADAR INPUT AND VISUALIZE :D
