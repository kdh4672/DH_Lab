import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import pickle
import torchvision

#데이터 전처리 방식을 지정한다.
transform = transforms.Compose([

        transforms.ToTensor(), # 데이터를 PyTorch의 Tensor 형식으로 바꾼다.
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # 픽셀값 0 ~ 1 -> -1 ~ 1
])
train_data = torchvision.datasets.ImageFolder(root='/home/daehyeon/hdd/deepfake_1st/fake/CW/20200819/cw-jungyun/dffs',
                                        )


# 데이터를 한번에 batch_size만큼만 가져오는 dataloader를 만든다.
dataloader = DataLoader(train_data, batch_size=1, shuffle=False)
