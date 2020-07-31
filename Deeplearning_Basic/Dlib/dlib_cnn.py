import cv2
import dlib
import time
import numpy as np
import copy
dlib.DLIB_USE_CUDA = True
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
weight = './mmod_human_face_detector.dat'

face_detector = dlib.cnn_face_detection_model_v1(weight)

# image = /home/daehyeon/hdd/High_Resolution/S001/L1/E01
# image = cv2.imread('/home/daehyeon/detectron2/test_image/1.jpg')

ALL = list(range(0, 68))
index = ALL
count = 1
while True:

    file_name = '/home/daehyeon/Dlib/many_people2.jpg'.format(count)
    print(file_name)
    # Create a HOG face detector using the built-in dlib class
    if count ==2:
        break



    # Load the image into an array
    image = cv2.imread(file_name)
    crop = copy.deepcopy(image)
    if image is None :
        print("no file like that name")
        break



    start = time.time()
    faces_cnn = face_detector(image, 1)


    count = count +1


    for face in faces_cnn:




        shape = predictor(image, face.rect)  # 얼굴에서 68개 점 찾기

        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points)
        list_points[:,1]
        x = min(face.rect.left(), min(list_points[:,0]))
        y = min(face.rect.top(), min(list_points[:,1]))
        w = max(face.rect.right() - x , max(list_points[:,0]) -x)
        h = max(face.rect.bottom() - y, max(list_points[:,1]) -y) #bounding box가 작아서 shape 끝점으로 대체
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        crop = crop[y:y+h,x:x+w]

        for i, pt in enumerate(list_points[index]):
            pt_pos = (pt[0], pt[1])
            cv2.circle(image, pt_pos, 2, (0, 255, 0), -1)
    print("I found {} faces in the file {}".format(len(faces_cnn), file_name))
    img_height, img_width = image.shape[:2]
    cv2.putText(image, "CNN", (img_width-50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,0,255), 2)

    # image 띄우기
    #cv2.imshow("{}".format(count), crop)
    cv2.imshow("{}".format(count), image)
    #cv2.imwrite("./dlib_cnn_crop/{}.jpg".format(count), crop)
    # image 저장
    cv2.imwrite("./dlib_cnn_result/t{}.jpg".format(count), image)
    end = time.time()
    print('걸린시간:', format(end - start, '.2f'))
    cv2.waitKey()
    cv2.destroyAllWindows()
