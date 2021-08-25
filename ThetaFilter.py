import cv2
import numpy as np

""" FFT class to properly visualize fft of gray image

"""

class thetaFilter:

    def __init__(self, image, z, c, n, t):
        self.image_gray = image

    def set_theta(self):
        self.Desv_Ang = int(input("ingresa el valor de desviasion angular: "))
        self.Tetha = int(input("ingresa el valor de tehta : "))

    def filtering(self):
        image_gray_fft = np.fft.fft2(self.image_gray)
        image_gray_fft_shift = np.fft.fftshift(image_gray_fft)

        # fft visualization
        image_gray_fft_mag = np.absolute(image_gray_fft_shift)
        image_fft_view = np.log(image_gray_fft_mag + 1)
        image_fft_view = image_fft_view / np.max(image_fft_view)

        # pre-computations
        num_rows, num_cols = (self.image_gray.shape[0], self.image_gray.shape[1])
        enum_rows = np.linspace(0, num_rows - 1, num_rows)
        enum_cols = np.linspace(0, num_cols - 1, num_cols)
        col_iter, row_iter = np.meshgrid(enum_cols, enum_rows)
        half_sizer = num_rows / 2 - 1  # here we assume num_rows = num_columns
        half_sizec = num_cols / 2 - 1

        band_pass_mask = np.zeros_like(self.image_gray)
        Ang_cut_off_low = self.Tetha - self.Desv_Ang  # int(freq_cut_off_low * half_size)
        Ang_cut_off_high = self.Tetha - self.Desv_Ang
        Tethab = ((180 * np.arctan2((row_iter - half_sizer), (col_iter - half_sizec)) / np.pi ) * -1) > Ang_cut_off_low
        Tethaa = ((180 * np.arctan2((row_iter - half_sizer), (col_iter - half_sizec)) / np.pi ) * - 1) < Ang_cut_off_high

        idx_bp = np.bitwise_and(Tethab, Tethaa)
        band_pass_mask[idx_bp] = 1

        # filtering via FFT
        mask = band_pass_mask  # * 255
        fft_filtered = image_gray_fft_shift * mask
        image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
        image_filtered = np.absolute(image_filtered)
        image_filtered /= np.max(image_filtered)

        #cv2.imshow("Original image", self.image_gray)
        #cv2.imshow("Filter frequency response", mask * 255)
       # cv2.imshow("Filtered image", image_filtered)
        #cv2.waitKey(0)




