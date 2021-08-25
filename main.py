from ThetaFilter import thetaFilter
import cv2
import sys
import os

""" Filtro de componenete con FFT de la componenetes en funcion de laorientacion Taller 2 

    main.py <path_to_image> <image_name>
"""

# Pulsar el botón verde en la barra superior para ejecutar el script.
if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Comprobar que la imagen es válida
    assert image is not None, "There is no image at {}".format(path_file)

    # Llamado de la función
    filt = thetaFilter(image_gray)
    # Solicitud de parametros desv angular y angulo Theta
    filt.set_theta()
    # Filtrado de la imagen
    filt.filtering()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/