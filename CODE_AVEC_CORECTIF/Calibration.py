import cv2
import numpy as np

'''
J'utiliser ce fichier comme test pour voir le rendu de la camera avec le correctif, tu peux lancer le code ici.
Cela t'afficheras l'image corrigée uniquement
'''
# Paramètres de calibration de la caméra (trouvés sur GitHub)
camera_matrix = np.array([
    [509.1810293948491, 0.0, 329.6996826114546],
    [0.0, 489.7219438561515, 243.26037641451043],
    [0.0, 0.0, 1.0]
])

dist_coeffs = np.array([
    [0.10313391355051804, -0.24657063652830105, -0.001003806785350075,
     -0.00046556297715377905, 0.1445780352338783]
])

#Ouvre la webcam
cap = cv2.VideoCapture(0)

#Lire une première frame pour récupérer les dimensions
ret, frame = cap.read()
if not ret:
    print("Erreur : impossible de lire la vidéo.")
    cap.release()
    exit()

h, w = frame.shape[:2]

#Calcul de la nouvelle matrice de caméra
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), 1, (w, h)
)

#Boucle de capture en direct
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Correction de la distorsion
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Affichage
    cv2.imshow("Flux vidéo corrigé", undistorted_frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Nettoyage
cap.release()
cv2.destroyAllWindows()
