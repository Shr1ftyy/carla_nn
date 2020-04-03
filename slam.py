import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

SCL_FACTOR = 2

class FastSlam(object):
    def __init__(self):
        pass
    def paint(image):
        # Initiate FAST object with default values
        fast = cv.ORB_create(1000)
        # Disable nonmaxSuppression
        #fast.setNonmaxSuppression(0)
        kp = fast.detect(image,None)
        #print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
        img3 = (cv.drawKeypoints(image, kp, None, color=(0,255,0)))
        img3 = cv.resize(img3, (int(np.shape(img3)[1]/SCL_FACTOR),int(np.shape(img3)[0]/SCL_FACTOR))) 

        return img3
