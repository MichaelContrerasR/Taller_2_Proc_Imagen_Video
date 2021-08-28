from ThetaFilter import thetaFilter
from metodo import thetaFilter2
import numpy as np
import cv2
import sys
import os

""" Filtrado con FFT de las componenetes en funcion de la orientacion Taller 2 

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

    # Llamado de funcion e ingreso de parametros para banco de filtros
    filtR_0 = thetaFilter2(image_gray, 23, 0)
    filtR_45 = thetaFilter2(image_gray, 23, 45)
    filtR_90 = thetaFilter2(image_gray, 23, 90)
    filtR_135 = thetaFilter2(image_gray, 23, 135)
    filtR_180 = thetaFilter2(image_gray, 23, 180)

    # Banco de Filtros 0, 45, 90, 135, 180 grados utilizando metodo filtering de la clase thetaFilter2
    ima_0, mask_0 = filtR_0.filtering()
    ima_45, mask_45 = filtR_45.filtering()
    ima_90, mask_90 = filtR_90.filtering()
    ima_135, mask_135 = filtR_135.filtering()
    ima_180, mask_180 = filtR_180.filtering()
    sint_ima = (mask_0.astype(np.float) * ima_0 + mask_45.astype(np.float) * ima_45 + mask_90.astype(np.float)
                * ima_90 + mask_135.astype(np.float)* ima_135 + mask_180.astype(np.float) * ima_180) / 5

    # Promediar y Sintetizar imagenes resultantes del banco de filtros
    sint_ima1 = ( ima_0 + ima_45 + ima_90 + ima_135 + ima_180)/5

    # Visualizacion de las imagenes resultantes
    cv2.imshow("Imagen filtrada 0 grados", ima_0)
    cv2.imshow("Imagen filtrada 45 grados", ima_45)
    cv2.imshow("Imagen filtrada 90 grados", ima_90)
    cv2.imshow("Imagen filtrada 135 grados", ima_135)
    cv2.imshow("Imagen filtrada 180 grados", ima_180)
    cv2.imshow("Imagen Original", image_gray)
    cv2.imshow("Imagen sintetizada con banco de filtros", sint_ima1)
    cv2.waitKey(0)

    # Llamado a la clase thetaFilter para filtrar por FFT con entrada de angulo tetha y delta theta por usuario
    filt = thetaFilter(image_gray)
    # Solicitud de parametros desv angular y angulo Theta
    filt.set_theta()
    # Filtrado de la imagen y etrega de la misma en angulo ingresado
    filt.filtering()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/