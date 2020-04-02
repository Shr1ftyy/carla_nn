from car_env import CarEnv
import sys
import glob 
import numpy as np 
import os
from PIL import Image 
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
IMG_DIR = None


# Prints controls into console
def showLogs():
	controls = car.get_control()   
	print(CAMERA_MEM[:])
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
	IMG_DIR = f"{DIRECTORY}images\\{timestamp}.png"

	stitch = np.concatenate(CAMERA_MEM[:])
	im = Image.fromarray(stitch)
	im.save(f"{timestamp}.png")
	print(IMG_DIR)
	#try:
	#except:
	#	 pass
	timestamp += 1
	# v = car.get_velocity()
	# kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

	return f"{throttle},{steer},{brake}"


# Main function
def main():
	global data
	print('initializing memory')
	for _ in sensors:
		CAMERA_MEM.append(None)

	sensors[0].listen(lambda image: processimg(image, 0))
	sensors[1].listen(lambda image: processimg(image, 1))
	sensors[2].listen(lambda image: processimg(image, 2))
	sensors[3].listen(lambda image: processimg(image, 3))

	if not os.path.exists(DIRECTORY):
		os.mkdir(DIRECTORY)
		print('made dir')

	data = open(f"{DIRECTORY}controls.txt", "w")
	print('initializing sensors')
	time.sleep(5)

	#Activates autopilot -> Data is collected while autopilot is running 
	car.set_autopilot(enabled=True)

	try:
		while True:
			showLogs()
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
