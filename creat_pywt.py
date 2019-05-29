import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pywt


data_normal = scipy.io.loadmat('/mnt/data/VGG_ultrasound/RF/spectrogram/RF3/all_rf/normal/2/RFD0035-DYG_20180619090536/11.mat')
data_normal = data_normal['normal_Array']
data_sig_normal = data_normal[0,:]
sampling_rate_normal = len(data_sig_normal) #采样频率
t_normal = np.arange(0,1,1.0/sampling_rate_normal)

data_ill = scipy.io.loadmat('/mnt/data/VGG_ultrasound/RF/spectrogram/RF3/all_rf/ill/3/RFD0079-LHK_20181016102727/51.mat')
data_ill = data_ill['ill_Array']
data_sig_ill = data_ill[0,:]
sampling_rate_ill = len(data_sig_ill) #采样频率
t_ill = np.arange(0,1,1.0/sampling_rate_ill)
wavename = "morl"  #pywt.wavelist()
totalscal = 256
fc = pywt.central_frequency(wavename)#中心频率
cparam = 2 * fc * totalscal
scales = cparam/np.arange(totalscal,1,-1)
[cwtmatr_normal, frequencies_normal] = pywt.cwt(data_sig_normal,scales,wavename,1.0/sampling_rate_normal)#连续小波变换
[cwtmatr_ill, frequencies_ill] = pywt.cwt(data_sig_ill,scales,wavename,1.0/sampling_rate_ill)#连续小波变换

print(cwtmatr_normal.shape)
plt.figure(figsize=(10, 10))
# plt.subplot(211)
# plt.contourf(t_normal, frequencies_normal, abs(cwtmatr_normal))
# plt.ylabel(u"freq(Hz)")
# plt.xlabel(u"time(s)")
# plt.imshow(abs(cwtmatr_normal))
# plt.subplot(212)
plt.contourf(t_ill, frequencies_ill, abs(cwtmatr_ill),cmap='gray')
plt.ylabel(u"freq(Hz)")
plt.xlabel(u"time(s)")
plt.subplots_adjust(hspace=0.4)
plt.savefig('/mnt/data/9.27/1.jpg')

plt.show()


