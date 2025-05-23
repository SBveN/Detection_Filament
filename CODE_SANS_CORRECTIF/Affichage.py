import cv2
import numpy as np
import Camera 
import Parametre 
import Plateau 
import Filament

cap = Camera.get_camera()
#pixels_cm =  21.5
initial_trackbar_vals = [102, 80, 20, 214]
count = 0
Parametre.initializeTrackbars(initial_trackbar_vals)

while True:
    success, img = cap.read()
    if not success:
        print("Erreur")
        break
    
    img = cv2.resize(img, (Parametre.width, Parametre.height))

    #Créer une image vide pour debuguage
    imgBlank = np.zeros((Parametre.height, Parametre.width, 3), np.uint8) 

    #Niveaux de gris, flou
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) 

    #Création des 2 thresholds
    thres = Parametre.valTrackbars() 
    imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1]) 

    #Création matrice pour Dilatation/Erosion
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2) 
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)  
 
    #Trouver les contours grâce avec des copies de l'image principale
    imgContours = img.copy() 
    imgBigContour = img.copy() 


    # Lecture des valeurs trackbar
    contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    biggest, _ = Plateau.biggestContour(contours)

    #Envoi l'image avec les contours haut et bas défini vers la fonction "Dim_Plateau", qui va pouvoir la traiter
    DimensionPlateau, dimensions_cm = Plateau.Dim_Plateau(imgThreshold, Parametre.pixels_cm)

    imgPlateauOnly = np.zeros_like(img)

    if biggest.size != 0:

        #Permet le bon affichage dans l'affichage de warp en réorganisant les points
        biggest = Camera.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 3)
        imgBigContour = Plateau.drawRectangle(imgBigContour,biggest,2)

        #engendre des points pour les warps
        pts1 = np.float32(biggest) 
        pts2 = np.float32([[0, 0],[Parametre.width, 0], [0, Parametre.height],[Parametre.width, Parametre.height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (Parametre.width, Parametre.height))
 
        #REMOVE 20 PIXELS FORM EACH SIDE
        imgWarpColored = imgWarpColored[5:imgWarpColored.shape[0] - 5, 5:imgWarpColored.shape[1] - 5]
        imgWarpColored = cv2.resize(imgWarpColored,(Parametre.width, Parametre.height))
        #facteur_conversion_px_mm = Utilitaire.pixels_cm * 10  # 1 cm = 10 mm
        resultat_filament = Filament.detecter_filament(imgWarpColored, Parametre.pixels_cm_warped)

        if resultat_filament:
            contour, diam_cm = resultat_filament

            # Dessin des contours
            cv2.drawContours(imgWarpColored, [contour], -1, (0, 0, 255), 2)

            # Trouver un point de référence pour afficher le texte (ici centre approx.)
            M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 50, 50  # fallback

        # Affichage du diamètre estimé
        cv2.putText(imgWarpColored, f"{diam_cm:.2f} cm",
                (cx - 35, cy + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgWarpGray = cv2.GaussianBlur(imgWarpGray, (5, 5), 0)
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)
        contours, _ = cv2.findContours(imgAdaptiveThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = 1      # Trop petit = bruit
        max_area = 5   # Trop grand = plusieurs objets collés ?

        # Image Array for Display
        imageArray = ([img,imgContours],
            [imgWarpColored, imgAdaptiveThre])
 
    else:
        imageArray = ([img,imgContours],
                      [imgBlank, imgBlank])

    #Detecte s'il y a un contour
    if DimensionPlateau.size !=0 and (dimensions_cm is not None):

        #Dessine les contours
        cv2.drawContours(imgContours, [DimensionPlateau], -1, (0, 255, 0), 3)
        x, y, w, h = dimensions_cm
        width_cm = w
        height_cm = h

        #Affiche les dimensions du plateau
        text = f"Largeur: {width_cm:.2f} cm, Hauteur: {height_cm:.2f} cm"
        text_position = (int(x), int(y + h + 275))
        cv2.putText(imgContours, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        #Si il ne trouve pas de contours
        print("Pas de contours détecter")

    # LABELS FOR DISPLAY
    labels = [["Original","Contours"],
              ["Warp Prespective","Adaptive Threshold"]]
 
    stackedImage = Camera.stockImg(imageArray,0.75,labels)
    cv2.imshow("Result",stackedImage)
 
    # SAVE IMAGE WHEN 's' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgWarpColored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1
        
    key = cv2.waitKey(1)
    if key == 27:  #Code ASCII de 'Échap'
        break

cap.release()
cv2.destroyAllWindows()