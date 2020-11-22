import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import shutil
from torch.optim import lr_scheduler
from ArcMarginProduct import *
import copy
import time
from torch.utils.tensorboard import SummaryWriter

# Tensorboard : set path
path = '../runs10/ksj0.25'
try :
    shutil.rmtree(path)
except:
    pass
writer = SummaryWriter(path)
# ---------------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 3
in_channel = 3
batch_size = 150
learning_rate = 0.0001
num_epochs = 5000

transforms = transforms.Compose([
        transforms.Resize((112,112)),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(), # 데이터를 PyTorch의 Tensor 형식으로 바꾼다.
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # 픽셀값 0 ~ 1 -> -1 ~ 1
])

#load data
total_dataset = torchvision.datasets.ImageFolder(root='/home/daehyeon/hdd/ksj0.25',
                                        transform=transforms,
                                        )

total_size = len(total_dataset)
train_size = int(np.ceil(total_size*0.99))
test_size = int(np.floor(total_size*0.01))

train_dataset, test_dataset = torch.utils.data.random_split(total_dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Tensorboard : dataloader 캡쳐

dataiter = iter(train_loader)
images, labels = dataiter.next()
img_grid = torchvision.utils.make_grid(images)
# matplotlib_imshow(img_grid, one_channel=True)
writer.add_image('face_images', img_grid)

# ---------------------- #

''' define model'''

model = models.resnet50()
model.fc = nn.Linear(2048  , 512)
# model.load_state_dict(torch.load(path +'.pth'))
model.to(device)
margin = ArcMarginProduct(in_feature=512,out_feature=num_classes)
# margin.load_state_dict(torch.load(path+'Margin.pth'))
margin.to(device)
nomargin = ArcMarginForTest(in_feature=512,out_feature=num_classes)

# Tensorboard : network graph 생성

# writer.add_graph(margin, (model(images.to(device)),labels.to(device)))
# writer.close()
classes = tuple([x for x in range(0,num_classes)])
# ---------------------- #

# Tensorboard : projector 생성
# data_list = torch.empty(0)
# label_list = torch.empty(0,dtype=int)
# for i in range(int(len(train_dataset)/1000)):
#     data_list = torch.cat((data_list,train_dataset.__getitem__(i)[0]),dim=0)
#     print(data_list.shape)
#     label_list = torch.cat((label_list,torch.tensor([train_dataset.__getitem__(i)[1]])),dim=0)
#     print(i)
# data_list = data_list.reshape(-1,3,112,112)
# def select_n_random(data, labels, n=100):
#     '''
#     데margin, (model(images.to(device)),labels.to(device))이터셋에서 n개의 임의의 데이터포인트(datapoint)와 그에 해당하는 라벨을 선택합니다
#     '''
#     assert len(data) == len(labels)
#
#     perm = torch.randperm(len(data))
#     return data[perm][:n], labels[perm][:n]
#
# images, labels = select_n_random(data_list, label_list)
#
# class_labels = [classes[lab] for lab in labels]
#
# features = images.view(-1, 3*128 * 128)
# writer.add_embedding(features,
#                     metadata=class_labels,
#                     label_img=images)
# writer.close()
# ---------------------- #

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'weight_decay': 5e-6},
    {'params': margin.parameters(), 'weight_decay': 5e-6}
], lr=learning_rate)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma= 0.99
)
breaker = False
if __name__ == '__main__':
    total_step = len(train_loader)


    for epoch in range(num_epochs):
        if breaker == True :
            break
        # if epoch % 20 == 0 and epoch != 0 :
        #     if epoch % 40 == 0 :
        #         print( "40:")
        #         for param in margin.parameters() :
        #                 param.requires_grad= False
        #         for param in model.parameters() :
        #                 param.requires_grad = True
        #     else :
        #         print( "20:")
        #         for param in model.parameters() :
        #                 param.requires_grad = False
        #         for param in margin.parameters() :
        #                 param.requires_grad= True
        #

        for i, (images, labels) in enumerate(train_loader):
            start = time.time()

            # Assign Tensors to Configured Device
            images = images.to(device)
            labels = labels.to(device)

            # Forward Propagation
            logits = model(images)
            outputs,degree,degree2 = margin(logits, labels)

            # Get Loss, Compute Gradient, Update Parameters
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# tesnorboard : loss 그래프 생성

            if i % 2 == 1:
                writer.add_scalar('intra_degree',
                            degree,
                            epoch * len(train_loader) + i)
                writer.add_scalar('inter_degree',
                            degree2,
                            epoch * len(train_loader) + i)
                writer.add_scalar('training loss',
                            loss.item(),
                            epoch * len(train_loader) + i)
                writer.add_scalar('learning_rate',
                            scheduler.get_last_lr()[0],
                            epoch * len(train_loader) + i)
            # if degree < 23:
            #     torch.save( model.state_dict() , path + '.pth' )
            #     torch.save( margin.state_dict() , path + 'Margin.pth' )
            #     breaker =True
            #     break
# ---------------------- #
            # Print Loss for Tracking Training
            if (i+1) % 1 == 0:
                print("learning_rate:{:.8f}".format(scheduler.get_last_lr()[0]))
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                nomargin.to(device)
                _, predicted = torch.max(outputs ,1)
                print('Testing data: [Predicted: {} / Real: {}]'.format(predicted, labels))
                nomargin.to('cpu')
            print("걸리는 시간: ", time.time()-start)


            if i % 2 == 1:
                model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        logits = model(images)
                        nomargin.weight = copy.deepcopy(margin.weight)
                        nomargin.to(device)
                        outputs = nomargin(logits)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        nomargin.to('cpu')
                    if epoch%13 == 12:
                        writer.add_scalar('accuracy',
                        100 * correct / total,
                        epoch * len(train_loader) + i)
                        print("===========================================================")
                        print('Accuracy of the network on the {} test images:\
                        {} %'.format(len(test_loader)*batch_size, 100 * correct / total))
                        print("===========================================================")
                model.train()
        if epoch %4 ==3:
            scheduler.step()
        torch.save(model.state_dict(), path+'.pth')
        torch.save(margin.state_dict(), path+'Margin.pth')

torch.save(model.state_dict(), path+'.pth')
torch.save(margin.state_dict(), path+'Margin.pth')

#
# ct = 0
# for param in model.parameters():
#     ct = ct + 1
#     if ct < 47:
#         param.requires_grad = True
#     else:
#         print(param)
