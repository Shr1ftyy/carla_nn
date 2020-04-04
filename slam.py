import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

SCL_FACTOR = 2

class ORBSlam(object):
    def __init__(self):
        pass
    def paint(image):
        # Initiate FAST object with default values
        orb1 = cv.ORB_create(3000)
        # Disable nonmaxSuppression
        #orb.setNonmaxSuppression(0)
        kp, des = orb.detectAndCompute(image,None)
        #print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
        img3 = (cv.drawKeypoints(image, kp, None, color=(0,255,0)))
        img3 = cv.resize(img3, (int(np.shape(img3)[1]/SCL_FACTOR),int(np.shape(img3)[0]/SCL_FACTOR))) 

        return img3
