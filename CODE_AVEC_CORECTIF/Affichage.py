import cv2
import numpy as np
import Camera
import Parametre 
import Plateau 
import Filament

# Appel de la caméra brut
cap_raw = Camera.get_camera()

# Appel de la caméra corrigée 
cap_corrected = Camera.correction_img(cap_raw)
use_corrected = False  # Choix caméra (raw ou corrigée)
count = 0
initial_trackbar_vals = [102, 80, 20, 214]
Parametre.initializeTrackbars(initial_trackbar_vals)

while True:
    # Choix du flux vidéo en fonction du booléen use_corrected
    print("Boucle démarrée, use_corrected =", use_corrected)
    if not use_corrected:
        success, img = cap_raw.read()
    else:
        success, img = cap_corrected.read()
    if not success:
        print("Erreur capture")
        break
    print("Image capturée")

    #Traitements de l'image brut
    img = cv2.resize(img, (Parametre.width, Parametre.height))
    imgBlank = np.zeros((Parametre.height, Parametre.width, 3), np.uint8) 
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) 
    thres = Parametre.valTrackbars() 
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1]) 
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) 
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  
    imgContours = img.copy() 
    imgBigContour = img.copy() 
    contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    biggest, _ = Plateau.biggestContour(contours)
    DimensionPlateau, dimensions_cm = Plateau.Dim_Plateau(imgThreshold, Parametre.pixels_cm)
    imgPlateauOnly = np.zeros_like(img)

    #Passage automatique à l'image corrigée si plateau détecté (seulement au départ)
    if not use_corrected:
        if biggest.size != 0:
            use_corrected = True
            print("Passage à image corrigée")
            
    #Création du contour du plateau
    if biggest.size != 0:
        biggest = Camera.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 3)
        imgBigContour = Plateau.drawRectangle(imgBigContour,biggest,2)

        #Prends des points de référence pour l'image warp (zoom sur le plateau une fois détecter)
        pts1 = np.float32(biggest) 
        pts2 = np.float32([[0, 0],[Parametre.width, 0], [0, Parametre.height],[Parametre.width, Parametre.height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (Parametre.width, Parametre.height))

        #Supprimer 5 pixels sur chaque bord (pour ne prendre que ce qu'il y a de présent au milieu du plateau en évitant ainsi les erreurs de luminosité)
        imgWarpColored = imgWarpColored[5:imgWarpColored.shape[0] - 5, 5:imgWarpColored.shape[1] - 5]

        #Traitements de l'image warp et de l'image adaptive threshold
        imgWarpColored = cv2.resize(imgWarpColored,(Parametre.width, Parametre.height))
        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgWarpGray = cv2.GaussianBlur(imgWarpGray, (5, 5), 0)
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)

        #Détecte le filament
        resultat_filament = Filament.detecter_filament(imgWarpColored, Parametre.pixels_cm_warped)

        #S'il existe, en créer ses contours
        if resultat_filament:
            contour, diam_cm = resultat_filament

            cv2.drawContours(imgWarpColored, [contour], -1, (0, 0, 255), 2)

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 50, 50

            cv2.putText(imgWarpColored, f"{diam_cm:.2f} cm",
                        (cx - 35, cy + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        '''
    Pour l'affichage ci-dessous, tu as la possibilité d'afficher aussi imgGray,imgTreshold,imgBigContour et imgWarpGray si tu le souhaites.

    Il te faudra modifier la taille des array ci-dessous et leurs donner ces formes la:

        ...
        imageArray = ([img,imgGray,imgThreshold,imgContours],
                      [imgBigContour,imgWarpColored, imgWarpGray,imgAdaptiveThre])
    else 
        imageArray = ([img,imgGray,imgThreshold,imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    Il te faudra aussi changer le label plus bas par :

    lables = [["Original","Gray","Threshold","Contours"],
              ["Biggest Contour","Warp Prespective","Warp Gray","Adaptive Threshold"]]
        '''

        imageArray = ([img,imgContours],
                      [imgWarpColored, imgAdaptiveThre])
    else:
        imageArray = ([img,imgContours],
                      [imgBlank, imgBlank])

    #Affiche la valeur des dimensions du plateau si != 0
    if DimensionPlateau.size != 0 and (dimensions_cm is not None):
        cv2.drawContours(imgContours, [DimensionPlateau], -1, (0, 255, 0), 3)
        x, y, w, h = dimensions_cm
        width_cm = w
        height_cm = h
        text = f"Largeur: {width_cm:.2f} cm, Hauteur: {height_cm:.2f} cm"
        text_position = (int(x), int(y + h + 275))
        cv2.putText(imgContours, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        print("Pas de contours détectés")

    labels = [["Original","Contours"],
              ["Warp Perspective","Adaptive Threshold"]]

    stackedImage = Camera.stockImg(imageArray, 0.75, labels)
    cv2.imshow("Result", stackedImage)
    key = cv2.waitKey(1) & 0xFF

    #Sauvegarde l'image du Warp
    if key == ord('s'):
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgWarpColored) #Il te faudra créer un fichier Scanned sur ton chemin d'où se situe ton code pour enregistrer les images que tu le souhaites
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1

    #Ferme le programme une que tu presses la touche "Echap" (il faut parfois maintenir 2-3 sec pour qu'il se ferme)
    elif key == 27:
        break

#"Relache" les images fournies par les caméras, pour les ré-utiliser dans un autre programme, pour bien arreter le code entre autre.
cap_raw.release()
if hasattr(cap_corrected, "cap"):
    cap_corrected.cap.release()
cv2.destroyAllWindows()
