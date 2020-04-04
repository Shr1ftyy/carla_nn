import cv2
import numpy as np 
import os
import time
import argparse
from orb import FastSlam
import utils

parser = argparse.ArgumentParser(description='plays images from a selected directory')
parser.add_argument('directory', metavar='directory', type=str, nargs='?', help='directory to parse images from')
parser.add_argument('slam', metavar='slam', type=str, nargs='?', help='insert yes slam, no for normal playback')

args = parser.parse_args()
_dir = args.directory 
files = os.listdir(_dir)
convFiles = utils.imgsort(files) 
WAITKEY = 100
SCL_FACTOR = 2

print(convFiles)

os.chdir(_dir)

if args.slam.lower() == 'yes':
    for d in convFiles:
        print(f'frame:{d}')
        img = cv2.imread(f'{d}.png', -1)
        img = cv2.resize(img, (int(np.shape(img)[1]),int(np.shape(img)[0]))) 
        slamImg = []
        imgSplit = np.split(img,4)

        for image in imgSplit:
            slamImg.append(FastSlam.paint(image))

        img = np.concatenate(slamImg[:])
        cv2.imshow(f'preview', img)
        cv2.waitKey(WAITKEY)

else:
    for d in convFiles:
        print(f'frame:{d}')
        img = cv2.imread(f'{d}.png', -1)  
        img = cv2.resize(img, (int(np.shape(img)[1]/SCL_FACTOR),int(np.shape(img)[0]/SCL_FACTOR))) 
        cv2.imshow(f'preview', img)
        cv2.waitKey(WAITKEY)


exit()
