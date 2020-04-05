import cv2
import numpy as np 
import os
import time
import argparse
import utils

parser = argparse.ArgumentParser(description='plays images from a selected directory')
parser.add_argument('directory', metavar='directory', type=str, nargs='?', help='directory to parse images from')
parser.add_argument('slam', metavar='slam', type=str, nargs='?', help='insert yes slam, no for normal playback')

args = parser.parse_args()
_dir = args.directory 
files = os.listdir(_dir)
print(files)
convFiles = utils.imgsort(files) 
WAITKEY = 10 
SCL_FACTOR = 2

print(convFiles)

os.chdir(_dir)

class Extractor(object):
    def __init__(self):
        self.orb = orb = cv2.ORB_create(2000)
        self.last = None

    def extract(self, img):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING) 
        #detection
        kps = self.orb.detect(img)
        #computes descripters
        kps, des = self.orb.compute(img, kps)

        cleanMatch = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                  kp1 = kps[m.queryIdx].pt
                  kp2 = self.last['kps'][m.trainIdx].pt
                  cleanMatch.append((kp1, kp2))
            #matches = zip([kps[m.queryIdx] for m in matches], [self.last['kps'][m.trainIdx]for m in matches])

        self.last = {'kps': kps, 'des': des}
        
        return cleanMatch

f = Extractor()

def process_frame(img):
    matches = f.extract(img)
    if matches is None:
        return img
    else:

        for p1, p2 in matches:
            u1, v1 = map(lambda x: int(round(x)), p1)
            u2, v2 = map(lambda x: int(round(x)), p2)
            cv2.circle(img, (u1, v1), color=(0,255,0),radius=3)
            cv2.line(img, (u1, v1),(u2, v2), color=(0,0,255))

        return img

for image in convFiles:
    img = process_frame(np.split(cv2.imread(image+'.png', -1), 4)[0])
    cv2.imshow('_', img)
    cv2.waitKey(WAITKEY)
