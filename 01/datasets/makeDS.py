import pickle
import sys
import cv2 as cv
import numpy as np

with open('hw_1_train.pickle', 'rb') as f:
    train = pickle.load(f)

#n = len(train['data'])
n = 10000
dimOld = 28
dimNew = 7
#wh = len(train['data'][0])
wh = dimNew * dimNew
print(n, wh, 1)

for i in range(n):
	#sys.stderr.write(str(i) + "\n")

	img = np.asarray(train['data'][i]).reshape((dimOld,dimOld))
	img = cv.resize(img, (dimNew, dimNew))
	#cv.imshow("Img", img)
	#cv.waitKey(100)
	
	img = img.reshape(dimNew * dimNew)

	for j in range(dimNew * dimNew):
		sys.stdout.write(str(img[j] / 256.0) + " ")

	if train['labels'][i] == 5.0:
		sys.stdout.write("1\n");
	else:
		sys.stdout.write("0\n");
