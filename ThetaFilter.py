import cv2
import numpy as np

""" Clase para realizar filtrado FFT en funcion de la orientacion solicitando entrada por teclado de angulo tetha y delta tetha 

"""

class thetaFilter:

    def __init__(self, image):
        self.image_gray = image

    def set_theta(self):
        self.Desv_Ang = int(input("ingresa el valor de desviasion angular: "))
        self.Tetha = int(input("ingresa el valor de tehta : "))

    def filtering(self):
        # Filtrado FFT de las componentes respecto a la orientacion
        image_gray_fft = np.fft.fft2(self.image_gray)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)
        image_gray_fft_mag = np.absolute(image_gray_fft_shift)

        # fft visualization
        image_gray_fft_mag = np.absolute(image_gray_fft_shift)
        image_fft_view = np.log(image_gray_fft_mag + 1)
        image_fft_view = image_fft_view / np.max(image_fft_view)

        # pre-computations
        num_rows, num_cols = (self.image_gray.shape[0], self.image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_sizer = num_rows / 2  # punto medio filas
        half_sizec = num_cols / 2  # punto medio columnas

        # Filtro pasabanda segun orientacion angular
        band_pass_mask1 = np.zeros_like(self.image_gray)
        idx_ang_low = 180 * (np.arctan2(row_iter - half_sizer, col_iter - half_sizec)) / np.pi + 180 > (
                    self.Tetha - self.Desv_Ang)
        idx_ang_high = 180 * (np.arctan2(row_iter - half_sizer, col_iter - half_sizec)) / np.pi + 180 < (
                self.Tetha + self.Desv_Ang)
        idx_bp = np.bitwise_and(idx_ang_low, idx_ang_high)
        idx_ang_low1 = 180 * (np.arctan2(row_iter - half_sizer, col_iter - half_sizec)) / np.pi + 180 > (
                self.Tetha + 180 - self.Desv_Ang)
        idx_ang_high1 = 180 * (np.arctan2(row_iter - half_sizer, col_iter - half_sizec)) / np.pi + 180 < (
                self.Tetha + 180 + self.Desv_Ang)
        idx_bp1 = np.bitwise_and(idx_ang_low1, idx_ang_high1)
        idx_bpf = np.bitwise_or(idx_bp, idx_bp1)
        band_pass_mask1[idx_bpf] = 1
        band_pass_mask1[int(half_sizer), int(half_sizec)] = 1

        # filtering via FFT
        mask = band_pass_mask1
        fft_filtered = image_gray_fft_shift * mask
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)

        # Visualizacion de las imagenes resultantes
        cv2.imshow("Imagen Original", self.image_gray)
        cv2.imshow("Mascara de Filtro angular", mask * 255)
        cv2.imshow("Imagen Filtrada", image_filtered)
        cv2.waitKey(0)




