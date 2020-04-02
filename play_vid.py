import cv2
import numpy as np 
import os
import time
from slam import FastSlam

_dir = "./data/test"
files = os.listdir(_dir)
convFiles = []

for i in range(0, len(files)):
    convFiles.append(int(files[i].split('.')[0]))

convFiles.sort(reverse=False)

for num in range(0, len(convFiles)):
    convFiles[num] = str(convFiles[num])

print(convFiles)

os.chdir(_dir)
for d in convFiles:
    print(f'frame:{d}')
    img = cv2.imread(f'{d}.png', -1)  
    img = imS = cv2.resize(img, (int(np.shape(img)[1]/2),int(np.shape(img)[0]/2))) 
    cv2.imshow(f'preview', FastSlam.paint(image=img))

    cv2.waitKey(33)

exit()
