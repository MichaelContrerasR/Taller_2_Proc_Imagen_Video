import cv2
import numpy as np
import os
import sys

""" FFT based filtering

    python fft_filtering.py <path_to_image> <image_name>
"""

if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_fft = np.fft.fft2(image_gray)
    image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

    # fft visualizacion
    image_gray_fft_mag = np.absolute(image_gray_fft_shift)
    image_fft_view = np.log(image_gray_fft_mag + 1)
    image_fft_view = image_fft_view / np.max(image_fft_view)

    # pre-computations
    num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
    enum_rows = np.linspace(0, num_rows - 1, num_rows)
    enum_cols = np.linspace(0, num_cols - 1, num_cols)
    col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
    half_sizer = num_rows / 2 - 1  # se calcula punto medio en longitud de filas
    half_sizec = num_cols / 2 - 1 # se calcula punto medio en longitud de Columnas por si la imagen no estotalmente cuadrada


    # mask fltro pasa banda usando limite superior e inferior angular en función de la orientación de las componentes.
    band_pass_mask = np.zeros_like(image_gray)
    Desv_Ang = int(input("ingresa el valor de desviasion angular: ")) # desviacion angular
    Tetha = int(input("ingresa el valor de tehta : "))  # angulo tetha
    Ang_cut_off_low = Tetha - Desv_Ang
    Ang_cut_off_high = Tetha + Desv_Ang

    Tethab = ((180 * np.arctan2((row_iter - half_sizer), (col_iter - half_sizec)) / np.pi) *-1) > Ang_cut_off_low
    Tethaa = ((180 * np.arctan2((row_iter - half_sizer), (col_iter - half_sizec)) / np.pi) *-1) < Ang_cut_off_high
    idx_bp = np.bitwise_and(Tethab, Tethaa)
    band_pass_mask[idx_bp] = 1

    # filtering via FFT
    #mask pasa banda garantizando un limite superior e inferior
    mask = band_pass_mask * 255
    fft_filtered = image_gray_fft_shift * mask
    image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
    image_filtered = np.absolute(image_filtered)
    image_filtered /= np.max(image_filtered)

    cv2.imshow("Imagen Original", image)
    cv2.imshow("Filtro angular", mask)
    cv2.imshow("Imagen Filtrada", image_filtered)
    cv2.imshow("Imagen Filtr", image_fft_view)

    cv2.waitKey(0)