#Privacy Protection by blurring the faces

#import matplotlib to read image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#import cv2

#import cascade of classifiers and gaussian filter
from skimage import data
from skimage.filters import gaussian
from skimage.feature import Cascade

#read the image and make a copy of it
#img = cv2.imread('class.jpg')
img = plt.imread('class.jpg')
image = img.copy()


#to get image flags
print(image.flags)

'''
To display the detected faces
'''

def showDetectedFaces(image,detected):
	plt.imshow(image)
	img_desc = plt.gca()
	plt.set_cmap('rainbow')
	plt.axis('off')

	for patch in detected:
		img_desc.add_patch(patches.Rectangle((patch['c'], patch['r']), patch['width'], patch['height'], fill=False, color='r', linewidth=2))
		
	plt.show()

#load the trained file from the module root
trained_file = data.lbp_frontal_face_cascade_filename()

#initialize the detector cascade
detector = Cascade(trained_file)

#detect the faces
detected = detector.detect_multi_scale(img=image, scale_factor=1.2, step_ratio=1, min_size=(20,20), max_size=(100,100))
#to display detected faces
showDetectedFaces(image,detected)

'''
To extract the face rectangle from the image using
the coordinates of the detected
'''

def getFace(d):
	
	#X & Y are the starting points of the face rectangle
	x,y = d['r'], d['c']

	#width & height of the face rectangle
	width,height = d['r']+d['width'], d['c']+d['height']

	#extract the detected face
	face = image[x:width, y:height]
	return face

'''
To merge the blurry images obtained to form the final resultant image
'''

def mergeBlurryFace(original,gaussian_image):
	
	x,y = d['r'], d['c']
	width,height = d['r']+d['width'], d['c']+d['height']
	original[x:width, y:height] = gaussian_image
	return original

#for each detected face
for d in detected:
	
	#obtain the face cropped from detected coordinates
	face = getFace(d)

	#apply Gaussian filter to the extracted faces
	gaussian_face = gaussian(face, multichannel=True, sigma=1)

	#merge the blurry face to our final image and display it
	resulting_image = mergeBlurryFace(image,gaussian_face)

'''
To display the resulting image
'''

def show_image(image,title):
	plt.imshow(image)
	plt.title(title)
	plt.axis('off')
	plt.show()

#Display the result
show_image(resulting_image,"Blurred Faces")