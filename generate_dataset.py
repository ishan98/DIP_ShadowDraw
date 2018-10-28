import cv2
import os
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
print(NoOfEdges, lenEdges, brEdges)

#getting BiCE discriptors corresponding to each patch of size 60*60 with 50% overlap
for n in range(NoOfImages):
	getEdgeImage = edges[n]
	for x in range(9):
		for y in range(9):
			image_patch = getEdgeImage[x*30:x*30+60:1, y*30:y*30+60:1]
			



#plt.subplot(121),plt.imshow(edges[0],cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(edges[1],cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([]) 
#plt.show()


