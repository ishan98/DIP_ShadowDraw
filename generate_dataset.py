import cv2
import os
import math
from scipy import ndimage, misc
import numpy as np
from matplotlib import pyplot as plt

#resizing the images
images = []
for root, dirnames, filenames in os.walk("./dataset"):
    for filename in filenames:
        filepath = os.path.join(root, filename)
        image = ndimage.imread(filepath, mode="L")
        image_resized = misc.imresize(image, (300, 300))
        images.append(image_resized)

[NoOfImages, length, breadth] = [len(images), len(images[0]), len(images[0][0])]

#getting the edges 
edges = []
for i in range(NoOfImages):
	getImage = images[i]
	getedges = cv2.Canny(getImage,100,200)
	edges.append(getedges)


[NoOfEdges, lenEdges, brEdges] = [len(edges), len(edges[0]), len(edges[0][0])]
#print(NoOfEdges, lenEdges, brEdges)

#getPatches = []
getImagePatch = []
sift = cv2.xfeatures2d.SIFT_create()
#getting BiCE discriptors corresponding to each patch of size 60*60 with 50% overlap
for n in range(NoOfImages):
	getEdgeImage = edges[n]
	for x in range(9):
		for y in range(9):
			image_patch = getEdgeImage[x*30:x*30+60:1, y*30:y*30+60:1]
			getImagePatch.append(image_patch)
			kp, des = sift.detectAndCompute(image_patch,None)
			print(len(des))

[NoOfPatch, lenPatch, brPatch] = [len(getImagePatch), len(getImagePatch[0]), len(getImagePatch[0][0])]

print(NoOfPatch, lenPatch, brPatch)

getpatch = getImagePatch[0]
sift = cv2.xfeatures2d.SIFT_create()

#[lenPatch, brPatch] = [len(getpatch), len(getpatch[0])]
#print(lenPatch, brPatch)
"""
gx = np.zeros(shape = (60,60))
gy = np.zeros(shape = (60,60))
gx_dash = np.zeros(shape = (60,60))
gy_dash = np.zeros(shape = (60,60))
theta = np.zeros(shape = (60,60))




#getting BiCE discriptor
pi = 3.14
for n in range(NoOfPatch):

	padded_patch = np.zeros(shape = (62,62))
	patch = getImagePatch[n]
	padded_patch[1:61,1:61] = patch[:,:]
	
	for x in range(lenPatch):
		for y in range(brPatch):
			gx[x,y] = padded_patch[x+1,y] - padded_patch[x,y]
			gy[x,y] = padded_patch[x,y+1] - padded_patch[x,y]
			theta[x,y] = math.atan2(gy[x,y],gx[x,y])
			c, s = np.cos(theta[x,y]), np.sin(theta[x,y])
			r = np.array(((c, -s), (s,c)))
			getxy = np.array(((x),(y)))
			gx_dash[x,y], gy_dash[x,y] = np.dot(r,getxy)
	
	getDiscriptors = np.zeros(shape=(18,6,4))
	x_factor = float(60)/float(18)
	y_factor = float(60)/float(6)
	theta_factor = float(2*pi)/float(4)
	
	for x in range(lenPatch):
		for y in range(brPatch):
			x_bar = int(math.floor(float(gx_dash[x,y])/float(x_factor)))
			y_bar = int(math.floor(float(gy_dash[x,y])/float(y_factor)))
			theta_bar = int(math.floor(float(theta[x,y]+pi)/float(theta_factor)))
			if(x_bar > 17):
				x_bar = 17
			if(y_bar > 5):
				y_bar = 5
			if(theta_bar > 3):
				theta_bar = 3
			if(x_bar < 0):
				x_bar = 0
			if(y_bar < 0):
				y_bar = 0
			if(theta_bar < 0):
				theta_bar = 0
			getDiscriptors[x_bar][y_bar][theta_bar] = getDiscriptors[x_bar][y_bar][theta_bar] + 1

"""


#print(count, ncount)





#plt.subplot(121),plt.imshow(edges[0],cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(edges[1],cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([]) 
#plt.show()


