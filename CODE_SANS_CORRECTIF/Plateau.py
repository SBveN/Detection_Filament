import cv2
import numpy as np
import Camera 

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area

#Dessine un rectangle
def drawRectangle(img,biggest,thickness):
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
 
    return img

def Dim_Plateau(imgCanny, pixels_cm):
    contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest = np.array([])  
    max_area = 0 
    dimensions_cm = None 

    for c in contours:
        area = cv2.contourArea(c)
        if area > 5000:  

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4 and area > max_area:
                biggest = approx
                max_area = area
                x, y, w, h = cv2.boundingRect(biggest)
                print(w, h)
                width_cm = round(w / pixels_cm, 2)
                height_cm = round(h / pixels_cm, 2)
                dimensions_cm = (x, y, width_cm, height_cm)
        
    
    if biggest.size != 0 and dimensions_cm is not None:
        return biggest, dimensions_cm
    else:
        return np.array([]), None
