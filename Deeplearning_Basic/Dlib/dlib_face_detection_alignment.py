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

device=torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
weight = './mmod_human_face_detector.dat'
detector = dlib.cnn_face_detection_model_v1(weight)
predictor = dlib.shape_predictor('./shape_predictor_5_face_landmarks.dat')
ALL = list(range(0, 5))
index = ALL

img_frame = cv.imread("./rotated_face2.jpg")
# img_frame = cv.resize(img_frame,dsize = (0,0), fx=1,fy=1)
# cv.imshow("dd",img_frame)
# cv.waitKey(0)
# cv.destroyAllWindows()
#     img_frame = cv.flip(img_frame, 0)  # 상하반전
gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)
#
dets = detector(gray, 1)
print("얼굴 갯수:", "{}개".format(len(dets)))
for face in dets:
    shape = predictor(gray, face.rect)
    list_points = []
    for p in shape.parts():
        list_points.append([p.x, p.y])
    list_points = np.array(list_points)
    for i, pt in enumerate(list_points[index]):
        pt_pos = (pt[0], pt[1])
        cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)
    nobox = copy.deepcopy(img_frame)
    fa = FaceAligner(predictor,desiredLeftEye=(0.3, 0.3), desiredFaceWidth=300)
    faceAligned = fa.align(img_frame,gray,face.rect)
    # cut = copy.deepcopy(faceAligned)
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y #bounding box가 작아서 shape 끝점으로 대체
    cv.rectangle(img_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    print("x:",x,"y:",y,"w:",w,"h:",h)
#
    cv.imwrite('face.jpg',faceAligned)

#
cv.imshow("algined",faceAligned)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow("result",img_frame)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow("nobox",nobox)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite("./before.jpg",img_frame)
cv.imwrite("./nobox.jpg",nobox)
cv.imwrite("./algined.jpg",faceAligned)

