import cv2
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
weight='./mmod_human_face_detector.dat'

detector=dlib.cnn_face_detection_model_v1( weight )

predictor=dlib.shape_predictor( './shape_predictor_68_face_landmarks.dat' )

img = cv2.imread('5.jpg')
cv2.waitKey()
# range는 끝값이 포함안됨
ALL=list( range( 0 , 68 ) )
RIGHT_EYEBROW=list( range( 17 , 22 ) )
LEFT_EYEBROW=list( range( 22 , 27 ) )
RIGHT_EYE=list( range( 36 , 42 ) )
LEFT_EYE=list( range( 42 , 48 ) )
NOSE=list( range( 27 , 36 ) )
MOUTH_OUTLINE=list( range( 48 , 61 ) )
MOUTH_INNER=list( range( 61 , 68 ) )
JAWLINE=list( range( 0 , 17 ) )

index=ALL
count=0

count=count + 1
print( count )
#    img_frame = cv.resize(img_frame,dsize = (0,0), cv2.INTER_CUBIC)
# img_frame=cv.flip( img_frame , 0 )  # 상하반전
#     cv.imshow("dd",img_frame)
# img_gray=cv.cvtColor( img_frame , cv.COLOR_BGR2GRAY )

dets=detector( img , 1 )
print( "faces:{}개".format( len( dets ) ) )
for face in dets :
  crop=copy.deepcopy( img )

  shape=predictor( img , face.rect )  # 얼굴에서 68개 점 찾기
  list_points=[ ]
  for p in shape.parts() :
    list_points.append( [ p.x , p.y ] )
  list_points=np.array( list_points )

  x=min( face.rect.left() , min( list_points[ : , 0 ] ) )
  y=min( face.rect.top() , min( list_points[ : , 1 ] ) )
  w=max( face.rect.right() - x , max( list_points[ : , 0 ] ) - x )
  h=max( face.rect.bottom() - y , max( list_points[ : , 1 ] ) - y )  # bounding box가 작아서 shape 끝점으로 대체
  cv.rectangle( img , (x , y) , (x + w , y + h) , (0 , 0 , 255) , 2 )

  crop=crop[ y :y + h , x :x + w ]

cv.imshow("img",img)
cv.imshow("crop",crop)
cv.waitKey()
key=cv.waitKey( 1 )

if key == 27 :
  pass

elif key == ord( '1' ) :
  index=ALL
elif key == ord( '2' ) :
  index=LEFT_EYEBROW + RIGHT_EYEBROW
elif key == ord( '3' ) :
  index=LEFT_EYE + RIGHT_EYE
elif key == ord( '4' ) :
  index=NOSE
elif key == ord( '5' ) :
  index=MOUTH_OUTLINE + MOUTH_INNER
elif key == ord( '6' ) :
  index=JAWLINE

cv.destroyAllWindows()

