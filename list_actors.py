
# @author Syeam_Bin_Abdullah _____ a script for playing around with Carla's API :D
import glob
import os
import sys

try:
    sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import *

import random
import time

client = carla.Client('localhost', 2069)
client.set_timeout(10)
world = client.get_world()
worldActors = world.get_actors()

for actor in worldActors:
    print(f'{type(actor)}')
    if str(type(actor)) == "<class 'carla.libcarla.Vehicle'>": 
        print(actor)

