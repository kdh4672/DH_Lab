import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
file_path = '/home/daehyeon/hdd/random_processed/'
list = os.listdir(file_path)
fake_count = 0
real_count = 0
for file in list:
	print("fake_count:",fake_count)
	print("real_count:",real_count)
	img = cv2.imread(file_path+file,0)
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)
	phase_spectrum = np.angle(fshift,deg=False)
	magnitude_spectrum = 20*np.log(np.abs(fshift))
	#
	rows,cols = img.shape
	crow,ccol = (int)(rows/2),(int)(cols/2)
	fshift[crow-15:crow+15, ccol-15:ccol+15] = 0
	#
	f_ishift = np.fft.ifftshift(fshift)
	img_back = np.fft.ifft2(f_ishift)
	img_back = np.abs(img_back)

	if file.split('_')[0] == 'fake':
		np.save('/home/daehyeon/hdd/numpy_files/phase_spectrum/fake_{}.npy'.format(fake_count),phase_spectrum.astype(np.float16))
		np.save('/home/daehyeon/hdd/numpy_files/magnitude_spectrum/fake_{}.npy'.format(fake_count),magnitude_spectrum.astype(np.float16))
		np.save('/home/daehyeon/hdd/numpy_files/img_back/fake_{}.npy'.format(fake_count),img_back.astype(np.float16))
		fake_count += 1
	else:
		np.save('/home/daehyeon/hdd/numpy_files/phase_spectrum/real_{}.npy'.format(real_count),phase_spectrum.astype(np.float16))
		np.save('/home/daehyeon/hdd/numpy_files/magnitude_spectrum/real_{}.npy'.format(real_count),magnitude_spectrum.astype(np.float16))
		np.save('/home/daehyeon/hdd/numpy_files/img_back/real_{}.npy'.format(real_count),img_back.astype(np.float16))
		real_count += 1