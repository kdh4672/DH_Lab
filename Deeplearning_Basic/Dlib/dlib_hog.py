import cv2
import dlib
import time


weight = './mmod_human_face_detector.dat'

face_detector = dlib.get_frontal_face_detector()


# image = cv2.imread('/home/daehyeon/detectron2/test_image/1.jpg')
#
# start = time.time()
#
# faces_hog = face_detector(image, 1)

count = 1
while True:

    file_name = '/home/daehyeon/detectron2/test_image/{}.jpg'.format(count)
    print(file_name)
    # Create a HOG face detector using the built-in dlib class




    # Load the image into an array
    try :
        image = cv2.imread(file_name)
    except:
        print("no file like that name")
        break
    start = time.time()

    faces_hog = face_detector(image, 1)

    end = time.time()
    count = count + 1
    print('걸린시간:',format(end-start,'.2f'))

    for face in faces_hog:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y

        cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)

    img_height, img_width = image.shape[:2]
    cv2.putText(image, "CNN", (img_width-50,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,0,255), 2)

    cv2.imshow("face detection with dlib", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
