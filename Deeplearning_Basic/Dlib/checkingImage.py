import cv2
import os
path = '/home/daehyeon/hdd/Three_Face/Kong'
file_list = os.listdir(path)
for file in file_list:
	im =cv2.imread(path+'/'+file)
	im = cv2.resize(im,dsize=(0,0),fx=3,fy=3)
	print(file)
	cv2.imshow("{}".format(file),im)
	cv2.waitKey()
	cv2.destroyAllWindows()
