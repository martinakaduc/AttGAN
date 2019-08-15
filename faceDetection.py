'''
Haar Cascade Face detection with OpenCV
    Based on tutorial by pythonprogramming.net
    Visit original post: https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/
Adapted by Marcelo Rovai - MJRoBot.org @ 7Feb2018
'''

import numpy as np
import cv2
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

def saveDetectedFace(path):
    img = cv2.imread(path)
    # print(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )
    newImg = []
    if (len(faces) < 1):
        newImg = img.copy()
    else:
        face = max(faces, key=lambda n: n[2]*n[3])
        x, y, w, h = face
        newImg = img[y:y+h, x:x+w]
    newPath = path[:-4] + '_' + path[-4:]
    cv2.imwrite(newPath, newImg)
    # print(newPath)
    return newPath
