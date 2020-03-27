# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
# @author Syeam_Bin_Abdullah ------->  A script  expandtab shiftwidth=4 softtabstop=4for playing around with Carla's API :D
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

PORT = 2069

client = carla.Client('localhost', PORT)
client.set_timeout(10)
world = client.get_world()

actor = world.get_actor(actor_id=180)
actor.destroy()
