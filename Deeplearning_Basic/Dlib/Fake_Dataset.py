import torch
import os
import cv2
import numpy as np
class FakeDataset(torch.utils.data.Dataset):
	def __init__(self,root_dir,transform=None):
		self.root_dir = root_dir
		self.list_ = sorted(os.listdir(self.root_dir))
		self.i_list , self.m_list , self.p_list= self.list_[ 0 ] , self.list_[ 1 ] , self.list_[ 2 ]
		self.list_m = os.listdir(self.root_dir+'/'+self.m_list)
	def __len__(self):
		return len(os.listdir(self.root_dir+'/'+self.m_list))

	def __getitem__(self, idx):
		file_name=self.list_m[idx]

		magnitude_spectrum= np.load( self.root_dir + '/' + self.m_list + '/' + file_name )
		phase_spectrum= np.load( self.root_dir + '/' + self.p_list + '/' + file_name )
		img_back= np.load( self.root_dir + '/' + self.i_list + '/' + file_name )

		mpi_stack=np.concatenate( ([ magnitude_spectrum ] , [ phase_spectrum ] , [ img_back ]) , axis=0 )
		pi_stack=np.concatenate( ([ phase_spectrum ] , [ img_back ]) , axis=0 )
		mp_stack=np.concatenate( ([ magnitude_spectrum ] , [ phase_spectrum ]) , axis=0 )
		only_p = [phase_spectrum]
		label = file_name.split('_')[0]

		if label == 'fake':
			label = 0
		else :
			label = 1

		return torch.Tensor(only_p),label

#
# list = os.listdir('/home/daehyeon/hdd/fft_pro/')
# print(str(list[1]).split('_')[0])
