#!C:/Users/Syeam/AppData/Local/Programs/Python/Python37/python.exe
import cv2 
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform, FundamentalMatrixTransform
import sys
import numpy as np 
import os
from itertools import compress
import time
import argparse
import utils
import pygame 
from pygame.locals import *
from car_env import CarEnv
from OpenGL.GL import *
from OpenGL.GLU import *

parser = argparse.ArgumentParser(description='plays images from a selected directory')
parser.add_argument('directory', metavar='directory', type=str, nargs='?', help='directory to parse images from')
parser.add_argument('slam', metavar='slam', type=str, nargs='?', help='insert yes slam, no for normal playback')

args = parser.parse_args()
_dir = args.directory 
files = os.listdir(_dir)
convFiles = utils.imgsort(files) 
WAITKEY = 500 
SCL_FACTOR = 2

# print(convFiles)

os.chdir(_dir)

points = (
        (-1, 0.5, 0),
        (1, 0.5, 0),
        (-1, -0.5, 0),
        (1, -0.5, 0),
        (-1, 0.5, -3),
        (1, 0.5, -3),
        (1, -0.5, -3),
        (-1, -0.5, -3),
        )

edges = (
        (0,1),
        (0,2),
        (0,4),
        (2,3),
        (3,1),
        (1,5),
        (6,3),
        (6,5),
        (4,5),
        (2,7),
        (4,7),
        (6,7),
        )

def add_ones(x):
  return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

class Extractor(object):
    def __init__(self, K):
        self.orb = orb = cv2.ORB_create(1000)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(K)

    # def normalize(self, pts):
    def normalize(self, pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]

    def denormalize(self, pt):
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        # print(ret)
        #ret /= ret[2]
        return int(round(ret[0])), int(round(ret[1]))


    def extract(self, img):
        """
        Extracts features using ORB and Essential Matrix from sequence of images using RANSAC
        """
        E = None
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING) 
        #detection
        kps = self.orb.detect(img)
        #computes descripters
        kps, des = self.orb.compute(img, kps)

        pts1 = []
        pts2 = []
        filterMatch = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            # Ratio test
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    pts1.append(kp1)
                    pts2.append(kp2)

            print(np.array(pts1).shape)
            print(np.array(pts1)[:10])
            pts1 = self.normalize(np.array(pts1))
            print('norm')
            print(pts1.shape)
            print(pts1[:10])
            pts2 = self.normalize(np.array(pts2))
            model, inliers = ransac((np.int32(pts1),
                np.int32(pts2)),
                EssentialMatrixTransform, 
                # FundamentalMatrixTransform, 
                min_samples=8,
                residual_threshold=0.1, max_trials=200)
            
            E = model.params
            pts1 = list(compress(pts1, inliers))
            pts2 = list(compress(pts2, inliers))

            for (i,j) in zip(pts1,pts2):
                filterMatch.append([i,j])
            # matches = zip([kps[m.queryIdx] for m in matches], [self.last['kps'][m.trainIdx]for m in matches])

        self.last = {'kps': kps, 'des': des}
       
        return np.array(filterMatch), E 

#Intrinsic Parameters
# foc = 20
foc = 1
H, W = np.split(cv2.imread(convFiles[0]+'.png', -1), 4)[0].shape[:-1]

K = np.array([[foc,  0,W//2],
             [   0,foc,H//2],
             [   0,  0,   1]])

f = Extractor(K)

f_est_avg = []

def EssentialtoRt(E):
    """
    Returns rotation and transformation from Essential Matrix
    """

    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    Winv = np.linalg.inv(W)
    Z = np.array([[0,1,0],[-1,0,0],[0,0,0]])
    # SVD on E
    U, S, Vt = np.linalg.svd(E)
    # S = np.diag(S)
    # print(f"S[0:2]: {S[0],S[1]}")

    #Weird estimation thing
    f_est = np.sqrt(2)/((S[0]+S[1])/2)
    f_est_avg.append(f_est)
    print(f_est, np.median(f_est_avg))

    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    # Rotation Matrix
    R = np.dot(np.dot(U, Winv),Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    # Transformation Matrix
    # print('ROTATION')
    # print(R)
    t = np.dot(np.dot(np.dot(U,W),S), U.T)
    # t = np.dot(np.dot(U,Z), U.T)

    return R, t

def Compute3D(y,y_p,R,t):
    """
    Compute 3D points from image points
    
    Parameters:
    y - normalized points from image 1
    y_p - normalized points from image 2
    R - rotation matrix
    t - translation matrix

    Returns:
    X - array of 3d points 

    """
    y.append(1)
    y_p.append(1)
    y = np.asarray(y)
    y_p = np.asarray(y_p)
    R = np.asarray(R)
    # t = np.diag(np.asarray(t)) FIX MAYBE

    r1 = R[0]
    r2 = R[1]
    r3 = R[2]


    # x3 = (np.dot(r1-(y_p[0]*r3),t))/(np.dot(r1-(y_p[0]*r3),y))
    x3 = np.dot(r1-np.dot(y_p[0], r3), t)/np.dot(r1-np.dot(y_p[0], r3), y)
    # print(x3.shape)
    # print(y[:2].shape)
    x1_2 = x3*(y[:2])

    # print("YO:")
    # print((round(x1_2[0]*W), round(x1_2[1]*H), round(x3*H)))
    return (x1_2[0], x1_2[1], x3)

def controls():
    events = pygame.event.get()
    keys = pygame.key.get_pressed()
    pressed_mouse = pygame.mouse.get_pressed()

    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit() 
            exit()

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4: # wheel rolled up
                glScaled(1.10, 1.10, 1.10)
            if event.button == 5: # wheel rolled down
                glScaled(0.9, 0.9, 0.9)

    if pressed_mouse[1]:
        ms = pygame.mouse.get_rel()
        glRotate(10, ms[1], ms[0], 0)

    if pressed_mouse[2]:
        ms = pygame.mouse.get_rel()
        glTranslatef(ms[0]/100, -1 * ms[1]/100, 0)


def render(pts, R, t):
    global points # Box of car LOL
    #Move car model around in 3D Space
    # print(R)
    # print(np.diag(t))

    # points = np.asarray(points) 
    # points = np.dot(np.dot(points,np.diag(t)),R)


    glEnable(GL_POINT_SMOOTH)
    glPointSize(1.5)

    glBegin(GL_POINTS)

    glColor3d(1,1,1)

    for point in pts:
        try:
            glVertex3d(point[0],point[1],point[2])
        except:
            pass

    glEnd()

    glEnable(GL_POINT_SMOOTH)
    glPointSize(10)
    glBegin(GL_POINTS)

    glColor3d(1,0,0)
    glVertex3d(0,0,0)
    # glVertex3d(points[0][0]-1,points[0][1]-0.5,points[0][2])


    glEnd()

    glBegin(GL_LINES)

    #Draws box for car
    for edge in edges:
        for point in edge:
            glColor3d(0,1,0)
            glVertex3fv(points[point])

    glEnd()

def process_frame(img):
    matches, E = f.extract(img)
    # print('ESSENTIAL') 
    # print(E) 
    pts = [] 
    if len(matches) < 1:
        return img, None, None, None
    else:
        # Perform Singular Value Decomposition on the current Essential matrix to derive rotation matrix R and transformation matrix T
        # print(np.linalg.det(E))
        R, t = EssentialtoRt(E)
        # R, t = None, None
        for p1, p2 in matches:
            # NORMALIZE? -- ignore, just using this tag as a bookmark :|
            # u1, v1 = map(lambda x: int(round(x)), p1)
            # u2, v2 = map(lambda x: int(round(x)), p2)
            u1,v1 = f.denormalize(p1)
            u2,v2 = f.denormalize(p2)
            cv2.circle(img, (u1, v1), color=(0,255,0),radius=1)
            cv2.line(img, (u1, v1),(u2, v2), color=(0,0,255))
            # Compute 3D points from image points, rotation and translation (normalized)
            pts.append(Compute3D([u1,v1], [u2,v2], R, t))

    return img, pts, R, t

yeet = False
def main():
    global yeet
    clock = pygame.time.Clock()
    clock.tick(60)
    pygame.init()
    display =  (1280, 720)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    gluPerspective(90, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0,0.0, -5)

    while True:

        try:
            for image in convFiles:
                img, pts, R, t = process_frame(np.split(cv2.imread(image+'.png', -1), 4)[0])
                glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
                if R is not None:
                    yeet = True

                if yeet:
                    try:
                        render(pts, R, t)
                    except Exception as e: 
                        raise e
                # print(R)
                # print(t)
                # car = np.array(points) 
                # print(car)
                # print(car.shape)
                pygame.display.flip()
                cv2.imshow('Preview - ORBSlam', img)
                cv2.waitKey(1)
                controls()

        except (KeyboardInterrupt, SystemExit):
            pygame.quit()
            sys.exit()
        


if __name__ == '__main__':
    main()
else:
    print('you cannot import the script... skipping import')

