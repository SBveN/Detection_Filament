import cv2
import numpy as np

#Défini la calibration de la Webcam Logitech C930e
def calibration_img():
    camera_matrix = np.array([
    [509.1810293948491, 0.0, 329.6996826114546],
    [0.0, 489.7219438561515, 243.26037641451043],
    [0.0, 0.0, 1.0]
    ])

    dist_coeffs = np.array([
    [0.10313391355051804, -0.24657063652830105, -0.001003806785350075,
     -0.00046556297715377905, 0.1445780352338783]
    ])
    
    return camera_matrix, dist_coeffs

#Récupère l'image brut de la caméra
def get_camera():
    cap = cv2.VideoCapture(0)
    # ... configuration de la caméra
    return cap

#Création d'une classe, qui permettra d'engendrer une image avec le correctif
class CorrectedCamera:
    def __init__(self, cap, mtx, dist):
        self.cap = cap
        self.mtx = mtx
        self.dist = dist

    def read(self):
        success, img = self.cap.read()
        if not success:
            return success, img
        h, w = img.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w,h), 1, (w,h))
        img_undistorted = cv2.undistort(img, self.mtx, self.dist, None, new_mtx)
        x, y, w_roi, h_roi = roi
        img_undistorted = img_undistorted[y:y+h_roi, x:x+w_roi]
        return success, img_undistorted

#Fonction qui enverra l'image corrigée 
def correction_img(base_cap):
    mtx = np.array([[600, 0, 320],
                    [0, 600, 240],
                    [0, 0, 1]], dtype=np.float32)
    dist = np.array([-0.2, 0.1, 0, 0, 0], dtype=np.float32)
    return CorrectedCamera(base_cap, mtx, dist)

#Mets tout dans une fenêtre et est super utile pour le deboguage
def stockImg(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

#ultra important pour l'affichage Warp ( pour mener à bien son affichage)
def reorder(myPoints):
 
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    
    return myPointsNew