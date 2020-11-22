import os
import re
import shutil
p = re.compile('[0-9]*')
for i in range(400):
	path = '/home/daehyeon/hdd/High_Crop/' +'{}'.format(i)
	file_list = os.listdir('/home/daehyeon/hdd/High_Crop/{}'.format(i))
	for file in file_list:
		m = p.match(file)
		if int(m.group())<36 or int(m.group()) > 40:
			os.remove(path+'/'+file)