import torch
import os
import cv2
import numpy as np
class FakeDataset(torch.utils.data.Dataset):
	def __init__(self,root_dir,normalize=False):
		self.root_dir = root_dir
		self.list_ = os.listdir(self.root_dir) ##list = fake, real
		self.normalize = normalize
	def __len__(self):
		return len(self.list_)

	def __getitem__(self, idx):
		img = np.load(self.root_dir + '/' +str(self.list_[idx]))
		if self.normalize == True:
			x = img
			x = (x - np.min(x)) / (np.max(x) - np.min(x))
			img = (x - 0.5) / 0.5

		name = str(self.list_[idx]).split('_')[0]
		if name == 'fake':
			name = 0
		else :
			name = 1
		number = str(self.list_[idx]).split('_')[1]
		return torch.Tensor(np.expand_dims(img, axis=0)),name

#
# list = os.listdir('/home/daehyeon/hdd/fft_pro/')
# print(str(list[1]).split('_')[0])
