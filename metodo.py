from ThetaFilter import thetaFilter
import cv2
import sys
import os

""" Manejo de imagen y creacion de Clase. Taller 1 

    main.py <path_to_image> <image_name>
"""

# Pulsar el botón verde en la barra superior para ejecutar el script.
if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    filt = thetaFilter(image_gray)
    filt.set_theta()
    filt.filtering()
