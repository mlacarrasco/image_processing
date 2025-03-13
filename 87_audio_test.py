from scipy.io import wavfile
import numpy as np
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt
#OJO: instalar pip install PyWavelets

Fs, x = wavfile.read('audio/flutecut.wav')


one_channel = x[:,0]
one_channel = one_channel/np.max(one_channel)

sigma = 0.05
data_noisy = one_channel +  sigma*np.random.randn(one_channel.size)

wavfile.write('audio/flute_noisy.wav',Fs, data_noisy)
data_denoise =  denoise_wavelet(data_noisy, 
                             method='VisuShrink', 
                             mode = 'soft',
                             wavelet='sym6',
                             wavelet_levels=6, 
                             rescale_sigma='True')

wavfile.write('audio/flute_denoisy.wav',Fs, data_denoise)


plt.figure()
plt.plot(data_noisy, label='noisy')
plt.plot(data_denoise, label='denoisy')
plt.legend()
plt.show()
