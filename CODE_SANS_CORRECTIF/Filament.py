import cv2
import numpy as np

def detecter_filament(image, pixels_cm_warped):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    meilleur_contour = None
    meilleur_diametre_cm = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50 or area > 3000:
            continue

        rect = cv2.minAreaRect(cnt)
        (center), (w, h), angle = rect
        diam_px = min(w, h)  # Le plus petit côté du rectangle englobant => épaisseur
        diam_cm = diam_px / (pixels_cm_warped * 1.89)

        if diam_cm > meilleur_diametre_cm:
            meilleur_diametre_cm = diam_cm
            meilleur_contour = cnt

    if meilleur_contour is not None:
        return meilleur_contour, meilleur_diametre_cm

    return None
