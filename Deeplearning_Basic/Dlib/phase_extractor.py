import os
file_path = '/home/capstone_ai1/kong/processed/fake/fake_256/'
list = os.listdir(file_path)
import cv2
import numpy as np
from matplotlib import pyplot as plt
count = 0

save_path = '/home/capstone_ai1/kong/fft_processed/fake/'

for file in list:
	count+=1
	if count < 26593:
		continue
	img = cv2.imread(file_path+file,0)
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)
	phase_spectrum = np.angle(fshift)
	np.save(save_path+'{}.npy'.format(count),phase_spectrum)
	if count == 30000 :
		break
	print(count)
