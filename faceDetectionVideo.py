#Privacy Protection by blurring the faces

#import OpenCV to read video
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

#import cascade of classifiers and gaussian filter
from skimage import data
from skimage.filters import gaussian
from skimage.feature import Cascade

#read the video
#use arg=0 for webcam
video_capture= cv2.VideoCapture(0)

'''
To display the detected faces
'''

def showDetectedFaces(image,detected):
	
	for patch in detected:
		print(patch['c'],patch['r'],patch['width'],patch['height'])
		cv2.rectangle(image,(patch['r'], patch['c']), (patch['r']+patch['width'], patch['c']+patch['height']), (255,0,0), 5)
	cv2.imshow("image",image)

''' using matplotlib
def showDetectedFaces(image,detected):
	plt.imshow(image)
	img_desc = plt.gca()
	plt.set_cmap('rainbow')
	plt.axis('off')

	for patch in detected:
		img_desc.add_patch(patches.Rectangle((patch['c'], patch['r']), patch['width'], patch['height'], fill=False, color='r', linewidth=2))
		
	plt.show()
'''	
		
#load the trained file from the module root
trained_file = data.lbp_frontal_face_cascade_filename()

#initialize the detector cascade
detector = Cascade(trained_file)

while (video_capture.isOpened()):

	ret, image = video_capture.read()
	image = cv2.resize(image,(640,480))

	#if not ret:
	#	break

	#detect the faces
	detected = detector.detect_multi_scale(img=image, scale_factor=1.3, step_ratio=1.2, min_size=(30,30), max_size=(600,450))
	#to display detected faces
	showDetectedFaces(image,detected)

	#Break the loop when user hits 'esc' key
	if cv2.waitKey(1) & 0xFF ==27:
		break

video_capture.release()
cv2.destroyAllWindows()
