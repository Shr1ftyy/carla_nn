import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os


def imgsort(files):
    """
    Sorts images from a directory into a list
    """
    convFiles = []
    for i in range(0, len(files)):
        convFiles.append(int(files[i].split('.')[0]))

    convFiles.sort(reverse=False)

    for num in range(0, len(convFiles)):
        convFiles[num] = str(convFiles[num])

    return convFiles

class FastSlam(object):
    """
    Class for leveraging FastSlam
    """
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor
        pass
    def paint(image):
        # Initiate FAST object with default values
        fast = cv.ORB_create(3000)
        # Disable nonmaxSuppression
        #fast.setNonmaxSuppression(0)
        kp= fast.detect(image,None)
        #print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
        img3 = (cv.drawKeypoints(image, kp, None, color=(0,255,0)))
        img3 = cv.resize(img3, (int(np.shape(img3)[1]/self.scale_factor),int(np.shape(img3)[0]/self.scale_factor))) 

        return img3
