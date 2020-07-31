# 학습 모델 다운로드
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
import dlib
import cv2 as cv
import numpy as np
weight = './mmod_human_face_detector.dat'

detector = dlib.cnn_face_detection_model_v1(weight)

predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

cap = cv.VideoCapture('./omg.mp4')

width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)



fourcc=cv.VideoWriter_fourcc( *'DIVX' )
out=cv.VideoWriter( 'output.mp4' , fourcc , 30 , (int( width ) , int( height )) )
print(width,height)

# range는 끝값이 포함안됨
ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))

index = ALL
count = 0




while (cap.isOpened()):
    count = count + 1
    print(count)
    ret, img_frame = cap.read()

    if ret == False :
        break;

#    img_frame = cv.resize(img_frame,dsize = (0,0), fx=0.6,fy=0.6)
    #img_frame = cv.flip(img_frame, 0)  # 상하반전
    img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)

    dets = detector(img_gray, 1)

    for face in dets:
        face = face.rect
        shape = predictor(img_frame, face)  # 얼굴에서 68개 점 찾기


        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points)

        for i, pt in enumerate(list_points[index]):
            pt_pos = (pt[0], pt[1])
            cv.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)

        cv.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()),
                     (0, 0, 255), 3)

    #cv.imshow('result', img_frame)

    out.write( img_frame )

    key = cv.waitKey(1)

    if key == 27:
        break

    elif key == ord('1'):
        index = ALL
    elif key == ord('2'):
        index = LEFT_EYEBROW + RIGHT_EYEBROW
    elif key == ord('3'):
        index = LEFT_EYE + RIGHT_EYE
    elif key == ord('4'):
        index = NOSE
    elif key == ord('5'):
        index = MOUTH_OUTLINE + MOUTH_INNER
    elif key == ord('6'):
        index = JAWLINE

cap.release()
out.release()
cv.destroyAllWindows()
