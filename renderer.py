import pygame 
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
import time
clock = pygame.time.Clock()

SCL_DOT = 1

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

points = list(vertices)
for point in points:
    points[points.index(point)] = (point[0]/SCL_DOT,point[1]/SCL_DOT,point[2]/SCL_DOT)

print(vertices)

def Cube():
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
    glEnd()

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
def main():
    clock.tick(60)
    pygame.init()
    display =  (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(120, (display[0]/display[1]), 0.1, 50.0)

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
            glRotate(0.3, ms[1], ms[0], 0)

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
        Cube()
        pygame.display.flip()

main()

