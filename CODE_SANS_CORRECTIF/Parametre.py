import cv2
import numpy as np

width, height = 640, 480 
width_Plat, height_Plat= 32, 30.5
pixels_cm =  9.92946
pixels_cm_warped_x = width / width_Plat
pixels_cm_warped_y = height / height_Plat
pixels_cm_warped = (pixels_cm_warped_x + pixels_cm_warped_y) / 2

#Permet de faire des changements de valeurs de trackbars
def initializeTrackbars(intialTracbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Threshold1", "Trackbars", 20, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 20, 255, nothing)
 
def valTrackbars():
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    src = Threshold1,Threshold2
    return src

def imgWarpColored(img, points, width, height):
    pts1 = np.float32(points.reshape(4, 2))
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_output = cv2.warpPerspective(img, matrix, (width, height))
    return img_output

def nothing(x):
    pass