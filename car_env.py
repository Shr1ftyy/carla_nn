import glob
import os
import sys

try:
            sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
                sys.version_info.major, 
                sys.version_info.minor,
                'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
        print("Failed to find carla's .egg directory")

import carla
from carla import *

import random
import time

#CONSTANTS
IMG_WIDTH = 640
IMG_HEIGHT = 480
FPS = 30 # --> sets maximum FPS for sensor inputs
TICKRATE = 1/FPS # --> sets sensor tickrate

class CarEnv:
    def __init__(self, port=2000):
        self.radartick = 1000 
        self.hfov = 70
        self.vfov = 70
        self.port = port
        self.im_width = IMG_WIDTH
        self.im_height = IMG_HEIGHT
        self.vehicle_list = [] # --> Vehicle List
        self.sensor_list = [] # --> Sensor List
        self.client = carla.Client('localhost', self.port)
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
        self.vehicle_list.append(self.vehicle)

        #SENSOR INIT
        self.front_rgb = self.blueprint_library.find('sensor.camera.rgb')
        self.front_rgb.set_attribute("sensor_tick", f"{TICKRATE}")
        self.front_rgb.set_attribute("image_size_x", f"{self.im_width}")
        self.front_rgb.set_attribute("image_size_y", f"{self.im_height}")
        self.front_rgb.set_attribute("fov", f"110")
        self.front_rgb.set_attribute("role_name", f"frontCam")

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("sensor_tick", f"{TICKRATE}")
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        #RADAR TO BE IMPLEMENTED FOR SEMANTIC SEGMENTATION AND POINT CLOUD IN THE FUTURE
        self.radar = self.blueprint_library.find('sensor.other.radar')
        self.radar.set_attribute("sensor_tick", f'{TICKRATE}')
        self.radar.set_attribute("horizontal_fov", f'{self.hfov}')
        self.radar.set_attribute("vertical_fov", f'{self.vfov}')
        self.radar.set_attribute("points_per_second", f'{self.radartick}')
        self.radar.set_attribute("range", f'30')

        self.frontTrans = carla.Transform(carla.Location(x=4, z=4))
        self.leftTrans = carla.Transform(carla.Location(x=2.5,y=-1, z=0.75), carla.Rotation(yaw=-90)) 
        self.rightTrans = carla.Transform(carla.Location(x=2.5, y=1, z=0.75), carla.Rotation(yaw=90))
        self.backTrans = carla.Transform(carla.Location(x=-2.5, z=0.75), carla.Rotation(yaw=180))


        #DEPTH

        #BGRA CAMERAS
        self.frontCam = self.world.spawn_actor(self.front_rgb, self.frontTrans, attach_to=self.vehicle_list[0])
        self.sensor_list.append(self.frontCam)
        self.leftCam = self.world.spawn_actor(self.rgb_cam, self.leftTrans, attach_to=self.vehicle_list[0])
        self.sensor_list.append(self.leftCam)
        self.rightCam = self.world.spawn_actor(self.rgb_cam, self.rightTrans, attach_to=self.vehicle_list[0])
        self.sensor_list.append(self.rightCam)
        self.backCam = self.world.spawn_actor(self.rgb_cam, self.backTrans, attach_to=self.vehicle_list[0])
        self.sensor_list.append(self.backCam)
    
        #RADAR
        self.frontRadar = self.world.spawn_actor(self.radar, self.frontTrans, attach_to=self.vehicle_list[0])
        self.sensor_list.append(self.frontRadar)

    def destroy(self):
        for car in self.vehicle_list:
            car.destroy()
        for sensor in self.sensor_list:
            sensor.destroy()
