import sys
import dlib
from skimage import io
import cv2
import numpy as np
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
# Take the image file name from the command line
count = 1
face_detector = dlib.get_frontal_face_detector()
ALL = list(range(0, 68))
index = ALL
while True:

    file_name = '/home/daehyeon/hdd/High_Resolution/S001/L1/E01/C{}.jpg'.format(count)
    print(file_name)
    # Create a HOG face detector using the built-in dlib class




    # Load the image into an array
    try :
        image = io.imread(file_name)
    except:
        print("no file like that name")
        break

    # Run the HOG face detector on the image data.
    # The result will be the bounding boxes of the faces in our image.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_faces = face_detector(image, 1)
    for face in detected_faces:

        shape = predictor(image, face)  # 얼굴에서 68개 점 찾기

        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points)

        for i, pt in enumerate(list_points[index]):
            pt_pos = (pt[0], pt[1])
            cv2.circle(image, pt_pos, 2, (0, 255, 0), -1)
    print("I found {} faces in the file {}".format(len(detected_faces), file_name))

    # Open a window on the desktop showing the image

    count = count +1
    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):
        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                                 face_rect.right(), face_rect.bottom()))
        cv2.rectangle(image, (face_rect.left(), face_rect.top()),(face_rect.right(), face_rect.bottom()),
                     (0, 0, 255), 3 )

    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

        # Draw a box around each face we found
