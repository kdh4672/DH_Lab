# 학습 모델 다운로드
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
import dlib
import cv2 as cv
import numpy as np
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
import os
from PIL import Image
from FaceAligner import FaceAligner
from Face_Rate import *
# f = open("/home/daehyeon/Dlib/lfw_class.txt",'r')
# classes = []
#
# while True:
#     data=f.readline()
#     classes.append(data.strip())
#     if not data:
#         break
#
# f.close()
#
# classes  = classes  + os.listdir('/home/daehyeon/hdd/Total_Face')

classes  = os.listdir('/home/daehyeon/hdd/High_Resolution_Files')+os.listdir('/home/daehyeon/hdd/Total_Face')
print(classes)
device=torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
num_classes= 403
path ='../easy_margin/403_300'
# path ='../runs3/266plus3'
model=models.resnet50()
model.fc=nn.Linear( 2048 , 512 )
model.load_state_dict( torch.load( path+ '.pth' ) )
model.to( device )
nomargin=ArcMarginForTest( in_feature=512 , out_feature=num_classes,easy_margin= True )
nomargin.load_state_dict( torch.load( path + 'Margin.pth' ) )
nomargin.to( device )
model.eval()
nomargin.eval()
print(model)



transforms = transforms.Compose([
        transforms.ToTensor(), # 데이터를 PyTorch의 Tensor 형식으로 바꾼다.
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # 픽셀값 0 ~ 1 -> -1 ~ 1
])

import os
# list_ = os.listdir('/home/daehyeon/hdd/Three_Face')
# list_ = os.listdir('/home/daehyeon/hdd/Test_Crop10')
# list_ = os.listdir('/home/daehyeon/hdd/face_crop')
# list_ = sorted(list_)
# print(list_)

weight = './mmod_human_face_detector.dat'

detector = dlib.cnn_face_detection_model_v1(weight)

predictor = dlib.shape_predictor('./shape_predictor_5_face_landmarks.dat')

cap = cv.VideoCapture('HS.mp4')
wanted = 'unknown'

width=int(cap.get( cv.CAP_PROP_FRAME_WIDTH ))
height=int(cap.get( cv.CAP_PROP_FRAME_HEIGHT ))



fourcc=cv.VideoWriter_fourcc( *'DIVX' )
# out=cv.VideoWriter( 'confidence.mp4' , fourcc , 60 , (int( width ) , int( height )) )
if wanted != 'June':
    r_width, r_height = int( width*0.5 ), int( height*0.5 )
else :
    r_height, r_width = int( width * 0.5 ) , int( height * 0.5 )

out=cv.VideoWriter( 'Sim_output.mp4' , fourcc , 25 , (r_width,r_height) )
out_cut = cv.VideoWriter('Test_cut.mp4', fourcc, 30 , (300,300))
print(width,height)

# range는 끝값이 포함안됨
ALL = list(range(0, 5))

index = ALL
count = 0
ct = 0
FA = 0
FR = 0
TP = 0
FP = 0
start = time.time()
while True:
    print("누적시간:",time.time()-start)
    if count == 250:
        break
    ret, img_frame = cap.read()

    if ret == False :
        break

    count = count + 1
    print("====================================")
    print("img 번호: ", count)
    img_frame = cv.resize(img_frame,dsize = (0,0), fx=1,fy=1)
    # matrix=cv.getRotationMatrix2D( (width / 2 , height / 2) , 90 , 1 )
    # img_frame=cv.warpAffine( img_frame , matrix , (width , height) )
    # img_frame = cv.flip(img_frame, 0)  # 상하반전
    gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)

    dets = detector(gray, 1)
    print("얼굴 갯수:", "{}개".format(len(dets)))
    for face in dets:
        fa = FaceAligner(predictor,desiredLeftEye=(0.3, 0.3), desiredFaceWidth=300)
        faceAligned = fa.align(img_frame,gray,face.rect)
        cut = copy.deepcopy(faceAligned)
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y #bounding box가 작아서 shape 끝점으로 대체
        cv.rectangle(img_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        print("x:",x,"y:",y,"w:",w,"h:",h)

        cv.imwrite('face.jpg',faceAligned)
        faceAligned = Image.open('face.jpg')
        # faceAligned = Image.fromarray(faceAligned)
        faceAligned = transforms(faceAligned).unsqueeze(0)


        predict,degree,nearest_degree = nomargin(model(faceAligned.to(device)))
        predicted_label = torch.max(predict, 1)[1]

        print("confidence1:",torch.max( predict , 1 )[ 0 ].item()/64)
        print("match 되나 확인:", classes[predicted_label])
        id = classes[predicted_label]
        if degree >50:
            id ='unknown'
            ct = ct+1
            print("Gap with W:",degree)
            cv.putText(img_frame, "{}".format("UnKnown"), (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                       (0, 0, 255), 2)
            cv.putText(img_frame, "Gap with W: {:.2f}".format(degree),
                       (x - 30, y - 40), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (255, 255, 0), 2)

        if wanted == 'unknown':
            if id == wanted:
                FR += 1
                print( "FR,FA:" , FR , FA )
                continue
            else:
                FA += 1
            print( "FR,FA:" , FR , FA )
        else:
            if id == wanted: ## Recall : TP/Total(TP+FN) Precision : TP/Total(TP+FP) FP ==0 이므로 계산 x
                TP += 1
                print("TP:",TP,"count:",count)



        cv.putText(img_frame, "{}".format(classes[predicted_label]), (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0, 0, 255), 2)

        confidence_list = 64*predict.to('cpu')/100

        softmax =nn.Softmax( dim=1 )
        s = softmax(predict/2)*100
        confidence = (round( torch.max( s , 1 )[ 0 ].item() , 2 ))
        print("confidence:",confidence)
        print("degree:","{:2f}도".format(degree))

        print("nearest_degree","{:2f}도".format(nearest_degree))
        # cv.putText( img_frame , "Confidence: {}%,degree: {:.2f},nearest_degree:{:.2f}".format(str( confidence),degree,nearest_degree) , (x-30 , y-40) , cv.FONT_HERSHEY_SIMPLEX , 1 ,
        #      (255 , 255 , 0) , 2 )
        cv.putText(img_frame, "Gap with W: {:.2f}".format(degree),
                   (x - 30, y - 40), cv.FONT_HERSHEY_SIMPLEX, 1,
                   (255, 255, 0), 2)

        print("label: ", classes[predicted_label])
        print("index: ", predicted_label)

    img_frame=cv.resize( img_frame , dsize = (0,0), fx=0.5,fy=0.5 )
    # cv.imshow( "result" , img_frame )
    # k = cv.waitKey(33)
    # if k==27:    # Esc key to stop
    #     break
    out.write( img_frame )
    # out_cut.write(cut)

face_rate = face_rate(FA,FR,count)
FAR = face_rate.FAR()
FRR = face_rate.FRR()
cap.release()
out.release()
out_cut.release()
cv.destroyAllWindows()
if wanted == "unknown":
    print( "FAR: " , FAR , "FRR: " , FRR )
else:
    print("Recall:", round(TP/count,3))



