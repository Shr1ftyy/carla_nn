from car_env import CarEnv
import sys
import glob 
import numpy as np 
import os

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

car_env = CarEnv()
car = car_env.vehicle_list[0]

#Activate autopilot (built-in carla function)
car.set_autopilot(enabled=True)


# Get controls applied by autopilot
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
    time.sleep(0.2)
