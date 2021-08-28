import cv2
import numpy as np
import os
import sys

""" Filtrado FFT en funcion de la orientacion de las componentes 

    python Cod_Funcional.py <path_to_image> <image_name>
"""

if __name__ == '__main__':
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    image = cv2.imread(path_file)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_fft = np.fft.fft2(image_gray)
    image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

    # Filtrado FFT de las componentes respecto a la orientacion
    image_gray_fft = np.fft.fft2(image_gray)
    image_gray_fft_shift = np.fft.fftshift(image_gray_fft)
    image_gray_fft_mag = np.absolute(image_gray_fft_shift)

    # fft visualization
    image_gray_fft_mag = np.absolute(image_gray_fft_shift)
    image_fft_view = np.log(image_gray_fft_mag + 1)
    image_fft_view = image_fft_view / np.max(image_fft_view)

    # parametros
    num_rows, num_cols = (image_gray.shape[0], image_gray.shape[1])
    enum_rows = np.linspace(0, num_rows - 1, num_rows)
    enum_cols = np.linspace(0, num_cols - 1, num_cols)
    col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
    half_sizer = num_rows / 2  # punto medio filas
    half_sizec = num_cols / 2  # punto medio columnas

    # Filtro pasabanda segun orientacion angular
    Desv_Ang = int(input("ingresa el valor de desviasion angular: "))
    Tetha = int(input("ingresa el valor de tehta : "))
    band_pass_mask1 = np.zeros_like(image_gray)
    idx_low = 180 * (np.arctan2(row_iter - half_sizer, col_iter - half_sizec)) / np.pi + 180 > (
                Tetha - Desv_Ang)
    idx_high = 180 * (np.arctan2(row_iter - half_sizer, col_iter - half_sizec)) / np.pi + 180 < (
            Tetha + Desv_Ang)
    idx_bp = np.bitwise_and(idx_low, idx_high)
    idx_low1 = 180 * (np.arctan2(row_iter - half_sizer, col_iter - half_sizec)) / np.pi + 180 > (
            Tetha + 180 - Desv_Ang)
    idx_high1 = 180 * (np.arctan2(row_iter - half_sizer, col_iter - half_sizec)) / np.pi + 180 < (
            Tetha + 180 + Desv_Ang)
    idx_bp1 = np.bitwise_and(idx_low1, idx_high1)
    idx_bpf = np.bitwise_or(idx_bp, idx_bp1)
    band_pass_mask1[idx_bpf] = 1
    band_pass_mask1[int(half_sizer), int(half_sizec)] = 1

    # filtrado via FFT
    mask = band_pass_mask1 * 255
    fft_filtered = image_gray_fft_shift * mask
    image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
    image_filtered = np.absolute(image_filtered)
    image_filtered /= np.max(image_filtered)

    # Visualizacion de las imagenes resultantes
    cv2.imshow("Imagen Original", image)
    cv2.imshow("Filtro angular", mask)
    cv2.imshow("Imagen Filtrada", image_filtered)
    cv2.imshow("Imagen Filtr", image_fft_view)

    cv2.waitKey(0)