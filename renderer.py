import pygame 
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from car_env import CarEnv
import numpy as np
import sys
import math

scl_pt = 1

RADAR_MEM = []
clock = pygame.time.Clock()

SCL_DOT = 1

detections = []

vertices = (
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

label_pts  = (
        (0,0,0),
        (3,0,0),
        (0,3,0),
        (0,0,3)
        )

labels = (
        (0,1),
        (0,2),
        (0,3),
        )

colors = [[1,0,0],[0,1,0],[0,0,1]]

points = list(vertices)
for point in points:
    points[points.index(point)] = (point[0]/SCL_DOT,point[1]/SCL_DOT,point[2]/SCL_DOT)

print(vertices)

def Render():

    glEnable(GL_POINT_SMOOTH)
    glPointSize(1.5)

    glBegin(GL_POINTS)

    glColor3d(1,1,1)

    for point in RADAR_MEM:
        try:
            glVertex3d(point[0],point[1],point[2])
        except:
            pass


    glEnd()

    glEnable(GL_POINT_SMOOTH)
    glPointSize(10)
    glBegin(GL_POINTS)

    glColor3d(1,0,0)
    glVertex3d(0, 0, 0)

    glEnd()

    glBegin(GL_LINES)

    for edge in edges:
        for point in edge:
            glColor3d(0,1,0)
            glVertex3fv(points[point])

    for label in labels:
        for pt in label:
            glColor3d(colors[labels.index(label)][0], colors[labels.index(label)][1], colors[labels.index(label)][2])
            glVertex3fv(label_pts[pt])

    glEnd()

def parse_data(radar_data):
    point = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
    if point.size != 0:
        print(point)
        point = np.reshape(point, (len(radar_data), 4))[0]
        depth = point[3]
        y = scl_pt*((depth)*(math.cos((point[1])+(math.pi/2))))   
        xz = scl_pt*(math.sqrt((depth**2)-(y**2)))
        x = scl_pt*((depth)*(math.sin(point[2])))
        z = scl_pt*((depth)*(math.cos(point[2])))
        # y = scl_pt*((depth)*(math.cos(math.pi - ((point[1])+(math.pi/2)))))   
        # if (point[1])+(math.pi/2) < 0:
        #     y = -1*y
        # xz = scl_pt*((depth)*(math.cos(abs(point[1]))))


        # x = scl_pt*((xz)*(math.cos(-1*(point[2]+(math.pi/2)))))
        # if -1*(point[2])+(math.pi/2) > math.pi/2:
        #     x = -1*x
        # z = scl_pt*((xz) * (math.sin(math.pi - (-1*(point[2])+(math.pi/2)))))

        point = np.array([x,y,z])
        
        RADAR_MEM.append(point)
    else:
        RADAR_MEM.append([0,0,100])

def backup():

    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                glRotate(10, -1, 0, 0)
            if event.key == pygame.K_DOWN:
                glRotate(10, 1, 0, 0)
            if event.key == pygame.K_LEFT:
                glRotate(10, 0, -1, 0)
            if event.key == pygame.K_RIGHT:
                glRotate(10, 0, 1, 0)
            if event.key == pygame.K_s:
                glTranslatef(0.0, 0.0, -3)
            if event.key == pygame.K_w:
                glTranslatef(0.0, 0.0, 3)

                if event.button == 3:
                    ms = pygame.mouse.get_rel()
                    mouse_position = pygame.mouse.get_pos()
                    glRotate(10, ms[1], ms[0], 0)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    glTranslatef(0,0,1)

                if event.button == 5:
                    glTranslatef(0,0,-1)

    point = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
    if point.size != 0:
        point = np.reshape(point, (len(radar_data), 4))[0]
        print(np.shape(point))
        depth = point[3]
        y = scl_pt*((depth)*(math.sin((point[1]))))   
        h = scl_pt*((depth)*(math.cos(point[1])))
        x = scl_pt*((h)*(math.sin(point[2])))
        z = scl_pt*((h)*(math.cos(point[2])))
        # x = scl_pt*((depth)*(math.sin(-1*(point[1])+(math.pi/2))*(math.cos(-1*(point[2])+(math.pi/2)))))
        # x = scl_pt*((depth)*(math.cos(-1*(point[2])+(math.pi/2))))
        # if -1*(point[2])+(math.pi/2) > math.pi/2:
        #     x = -1*x
        # z = scl_pt*(math.sqrt(-1*(x**2+y**2-depth**2)))
        # z = scl_pt*(depth*(math.sin(-1*(point[2])+(math.pi/2))))
        point = np.array([x,y,z])
        
        RADAR_MEM.append(point)
        print(point)
    else:
        RADAR_MEM.append([0,0,100])

def main():
    try:
        global RADAR_MEM
        car_env = CarEnv(port=2069)
        car = car_env.vehicle_list[0]
        sensors = car_env.sensor_list
        car.set_autopilot(enabled=True)
        HFOV = (car_env.hfov*math.pi)/180
        VFOV = (car_env.vfov*math.pi)/180

        for sensor in sensors:
            if sensor.type_id == 'sensor.other.radar':
                sensor.listen(lambda data: parse_data(data))

        clock.tick(60)
        pygame.init()
        display =  (1280, 720)
        pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

        gluPerspective(120, (display[0]/display[1]), 0.1, 200.0)

        glRotate(0, 0, 0, 0)
        glTranslatef(0.0, 0.0, -3)

        while True:
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
                glRotate(2, ms[1], ms[0], 0)

            if pressed_mouse[2]:
                ms = pygame.mouse.get_rel()
                glTranslatef(ms[0]/100, -1 * ms[1]/100, 0)

            if keys[pygame.K_UP]:
                glRotate(0.1, -1, 0, 0)
            if keys[pygame.K_DOWN]:
                glRotate(0.1, 1, 0, 0)
            if keys[pygame.K_LEFT]:
                glRotate(0.1, 0, -1, 0)
            if keys[pygame.K_RIGHT]:
                glRotate(0.1, 0, 1, 0)
            if keys[pygame.K_s]:
                glTranslatef(0.0, 0.0, -1)
            if keys[pygame.K_w]:
                glTranslatef(0.0, 0.0, 1)
                   

            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            Render()
            pygame.display.flip()

            if len(RADAR_MEM) >= car_env.radartick:
                RADAR_MEM = []

    except (KeyboardInterrupt, SystemExit):
        car.destroy()
        for sensor in sensors:
            sensor.destroy()
        pygame.quit()
        sys.exit()
        exit()

if __name__ == '__main__':
    main()
else:
    print('you cannot import the renderer... skipping import')

