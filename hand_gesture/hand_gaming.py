import cv2
import numpy as np
import urllib
import math

url = "http://192.168.137.31:8080/shot.jpg"
#capture = cv2.VideoCapture(0)
#lower = np.array([0, 48, 80], dtype = "uint8")
#upper = np.array([20, 255, 255], dtype = "uint8")
while True:
	#ret, frame = capture.read()
	img_res = urllib.urlopen(url)
	img_array = np.array(bytearray(img_res.read()),dtype=np.uint8)
	frame = cv2.imdecode(img_array,-1)
	#frame = cv2.imread("test\kalia.jpg",1)
	cv2.rectangle(frame,(50,150),(300,400),(0,255,0),1)
	#cv2.imshow("frame",frame)
	img = frame[150:400,50:300]
	#hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	#mask = cv2.inRange(hsv,lower,upper)
	#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
	#mask = cv2.erode(mask, kernel, iterations = 2)
	#mask = cv2.dilate(mask, kernel, iterations = 2)
 
	#mask = cv2.GaussianBlur(mask, (3, 3), 0)
	#skin = cv2.bitwise_and(img, img, mask = mask)
 	#cv2.imshow("skin",skin)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	#cv2.imshow("img",img)
	
	smooth = cv2.GaussianBlur(gray,(5,5),0)	
	#cv2.imshow("smooth",smooth)
	ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	#thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv2.THRESH_BINARY,11,2)


	(version,_,_,) = cv2.__version__.split('.')
	if version == '3':
		image,contours,heirarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	elif version == '2':	
		contours,heirarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#cv2.drawContours(frame,contours,-1,(0,0,255),3)
	
	count = max(contours, key = lambda x:cv2.contourArea(x))

	x,y,w,h = cv2.boundingRect(count)

	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

	hull = cv2.convexHull(count,returnPoints = False)
	defects = cv2.convexityDefects(count,hull)
	count_defects = 0
	if defects is None:
		defects = [[0,0,0,0]]

	else:
		for i in range(defects.shape[0]):
			s,e,f,d = defects[i,0]

			start = tuple(count[s][0])
			end = tuple(count[e][0])
			far = tuple(count[f][0])
	
			a = math.sqrt(((end[0]-start[0])**2) + ((end[1]-start[1])**2))
			b = math.sqrt(((far[0]-start[0])**2) + ((far[1]-start[1])**2))
			c = math.sqrt(((end[0]-far[0])**2) + ((end[1]-far[1])**2))
			#print a,b,c

			angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57 
			

			if angle<=90:
				count_defects+=1
				cv2.circle(img,far,5,(255,0,0),-1)
			cv2.line(img,start,end,(0,255,0),2)
		
		if count_defects == 1:
			cv2.putText(frame,"I am Akshat",(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,2)
		elif count_defects == 2:
			str = "this is a basic hand gesture recognizer"
			cv2.putText(frame,str,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,2)
   		elif count_defects == 3:
			cv2.putText(frame,"this is 4",(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,2)
	    	elif count_defects == 4:
			cv2.putText(frame,"Hey!!",(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,2)
		else:
			cv2.putText(frame,"Welcome",(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,2)

	#drawing = np.zeros(img.shape,np.uint8) 
	#cv2.drawContours(drawing,[count],0,(0,255,0),0)
	#cv2.drawContours(drawing,[hull],0,(0,255,0),0)
	cv2.imshow("thresh",thresh)
	cv2.imshow("img",img)	
	cv2.imshow("frame",frame) 
	k=cv2.waitKey(10)
	if k==27:
		break
#capture.release()
cv2.destroyAllWindows()
