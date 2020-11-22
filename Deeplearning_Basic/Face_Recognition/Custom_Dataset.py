import torch
import os
import cv2
class CustomDataset(torch.utils.data.Dataset):
	def __init__(self,root_dir):
		self.root_dir = root_dir
		self.list = os.listdir(self.root_dir)

	def __len__(self):
		return len(self.list)

	def __getitem__(self, idx):
		img = cv2.imread(self.root_dir + '/' +str(self.list_[idx]))
		name = str(self.list_[idx]).spli('_')[0]
		number = str(self.list_[idx]).spli('_')[1]
		return img,name

dataset = CustomDataset('/ home / daehyeon / hdd / Custom')