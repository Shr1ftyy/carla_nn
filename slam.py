import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

class FastSlam(object):
    def __init__(self):
        pass
    def paint(image):
        img = image
        # Initiate FAST object with default values
        fast = cv.FastFeatureDetector_create()
        # Disable nonmaxSuppression
        fast.setNonmaxSuppression(0)
        # find and draw the keypoints
        kp = fast.detect(img,None)
        #print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
        img3 = cv.drawKeypoints(img, kp, None, color=(0,255,0))

        return img3
