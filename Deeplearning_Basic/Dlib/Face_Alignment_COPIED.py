import cv2
import dlib
import time
import numpy as np
import copy
import os
from FaceAligner import FaceAligner

# predictor=dlib.shape_predictor( './shape_predictor_68_face_landmarks2.dat' )
predictor=dlib.shape_predictor( './shape_predictor_5_face_landmarks.dat' )

weight='./mmod_human_face_detector.dat'


face_detector=dlib.cnn_face_detection_model_v1( weight )
# file_name = '5.jpg'
# image = cv2.imread(file_name)
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
count=1

path = '/home/daehyeon/hdd/High_Resolution_Files'
folder_list = os.listdir(path)
print(folder_list)
for folder in folder_list:
    folder_path = path + '/' +folder +'/S001'  # '/home/daehyeon/hdd/High_Resolution_Files'
    file_list = [folder_path+'/L1/E01',folder_path+'/L2/E01']
    ct=0
    for file in file_list:  # file = folder_path+'/L1/E01',folder_path+'/L2/E01'

        for i in range(1,21):
            image = cv2.imread(file+'/C{}.jpg'.format(i))
            # image = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5)

# Create a HOG face detector using the built-in dlib class
# Load the image into an array

            start = time.time()
            # try:
            faces_cnn=face_detector( image , 1 )
            # except: continue
            if len( faces_cnn ) > 1:
                continue
            count=count + 1
            crop=copy.deepcopy( image )
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            for face in faces_cnn :
                shape=predictor( image , face.rect )  # 얼굴에서 68개 점 찾기
                list_points = []
                for p in shape.parts() :
                    list_points.append( [ p.x , p.y ] )
                list_points=np.array( list_points )

                x=min( face.rect.left() , min( list_points[ : , 0 ] ) )
                y=min( face.rect.top() , min( list_points[ : , 1 ] ) )
                w=max( face.rect.right() - x , max( list_points[ : , 0 ] ) - x )
                h=max( face.rect.bottom() - y , max( list_points[ : , 1 ] ) - y )  # bounding box가 작아서 shape 끝점으로 대체
                # cv2.rectangle( image , (x , y) , (x + w , y + h) , (0 , 0 , 255) , 2 )

                crop=crop[ y :y + h , x :x + w ]

                fa = FaceAligner(predictor,desiredLeftEye=(0.3, 0.3), desiredFaceWidth=300)
                faceAligned = fa.align(image,gray,face.rect)






                for i , pt in enumerate( list_points[ 0:3 ] ) :
                    pt_pos=(pt[ 0 ] , pt[ 1 ])
                    cv2.circle( image , pt_pos , 2 , (0 , 255 , 0) , -1 )

            print( "I found {} faces in the file {}".format( len( faces_cnn ) , file+'C{}.jpg'.format(i) ) )
            img_height , img_width=image.shape[ :2 ]
            # cv2.putText( image , "CNN" , (img_width - 50 , 40) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 0 , 255) , 2 )

            # cv2.imshow( "face detection with dlib" , image )
            # cv2.imshow("Aligned", faceAligned)
            # cv2.imshow( "Crop",crop)
            end=time.time()
            print( '걸린시간:' , format( end - start , '.2f' ) )
            new_path =  '/home/daehyeon/hdd/403_High/'
            cv2.imwrite(new_path+'{}_{}.jpg'.format(folder,ct),faceAligned)
            ct += 1
            cv2.waitKey()
            cv2.destroyAllWindows()
            print(dlib.DLIB_USE_CUDA)

