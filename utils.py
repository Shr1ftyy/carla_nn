import math
import os

def imgsort(files):
    convFiles = []
    for i in range(0, len(files)):
        convFiles.append(int(files[i].split('.')[0]))

    convFiles.sort(reverse=False)

    for num in range(0, len(convFiles)):
        convFiles[num] = str(convFiles[num])

    return convFiles

