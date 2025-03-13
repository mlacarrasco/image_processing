import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio

#lectura 
img = cv2.imread('imagenes/cameraman.png',0)

#generamos la imagen con ruido
img_norm = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
noise = np.random.random((img.shape))
img_noisy = cv2.addWeighted(img_norm, 0.85,  noise, 0.15, 0.0)

#estimacion señal/ruido medida en dB
sigma_est = estimate_sigma(img_noisy, average_sigmas=True)

#revisar tipos de wavelets
# http://wavelets.pybytes.com/
img_visushrink_haar= denoise_wavelet(
    img_noisy, 
    method='VisuShrink',
    mode='soft', 
    sigma = sigma_est/4,
    wavelet_levels = 4, 
    wavelet='haar', 
    rescale_sigma = True
    )

img_visushrink_rbio= denoise_wavelet(
    img_noisy, 
    method='VisuShrink',
    mode='soft', 
    sigma = sigma_est/4,
    wavelet_levels = 4, 
    wavelet='rbio1.5', 
    rescale_sigma = True
    )

img_visushrink_sym= denoise_wavelet(
    img_noisy, 
    method='VisuShrink',
    mode='soft', 
    sigma = sigma_est/4,
    wavelet_levels = 4, 
    wavelet='sym6', 
    rescale_sigma = True
    )

psnr_noisy = peak_signal_noise_ratio(img_norm, img_noisy)

print('Ruido estimado {:1.5f}'.format(sigma_est))
print('PSNR {:1.5f}'.format(psnr_noisy))

plt.figure()
plt.subplot(2,2,1)
plt.imshow(img_noisy, cmap='gray')
plt.title('Ruido estimado {:.5f} dB'.format(psnr_noisy))

plt.subplot(2,2,2)
plt.imshow(img_visushrink_haar, cmap='gray')
plt.title('Wavelet haar')

plt.subplot(2,2,3)
plt.imshow(img_visushrink_rbio, cmap='gray')
plt.title('Wavelet Rbio1.3')

plt.subplot(2,2,4)
plt.imshow(img_visushrink_sym, cmap='gray')
plt.title('Wavelet Sym4')
plt.show()