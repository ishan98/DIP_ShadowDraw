from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import os
import math
from scipy import ndimage, misc
import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial


def drawMatches(img1, kp1, img2, kp2, matches):

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    out[:rows1,:cols1] = np.dstack([img1])
    out[:rows2,cols1:] = np.dstack([img2])
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0, 1), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0, 1), 1)
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0, 1), 1)

    return out


def extract_features(image, alg, vector_size=32):
	#print('Hello')
	try:
		
		#alg = cv2.xfeatures2d.SIFT_create()
		kps = alg.detect(image)
		kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
		kps, dsc = alg.compute(image, kps)
		if(len(kps) != 0):
			dsc = dsc.flatten()
			needed_size = (vector_size * 64)
			if dsc.size < needed_size:
				dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
			dsc = dsc[:2048]
			#print(dsc)
		else:
			return(np.zeros(vector_size * 64))
	except cv2.error as e:
		print('Error: ', e)
		return None
	idx = dsc.argsort()[-1:-411:-1]
	binarize_dsc = np.zeros(2048)
	binarize_dsc[idx[:]] = 1
	return(binarize_dsc)


def getFeatureMap(img, sift):
	#alg = cv2.KAZE_create()
	kp, des = sift.detectAndCompute(img,None)
	temp_vector = np.zeros(128)
	if(len(kp) != 0):
		for x in range(len(des)):
			for y in range(128):
				temp_vector[y] = des[x][y]
	#print(temp_vector)
	return(temp_vector)

def getMatch(feature_map1,feature_map2):
	#intersection_cardinality = len(set.intersection(*[set(feature_map1), set(feature_map2)]))
	#union_cardinality = len(set.union(*[set(feature_map1), set(feature_map2)]))
	#return intersection_cardinality/float(union_cardinality)

	match_value = 0
	result = spatial.distance.cosine(feature_map1, feature_map2)
	return(result)
	
	#for x in range(len(feature_map1)):
	#	match_value = match_value + abs(feature_map1[x] - feature_map2[x])
	#return(match_value)

def make_bag_of_words():
	#resizing the images
	imdataset = []
	for root, dirnames, filenames in os.walk("./dataset/image_dataset"):
		for filename in filenames:
			filepath = os.path.join(root, filename)
			imdata = ndimage.imread(filepath, mode="L")
			imdata_resized = misc.imresize(imdata, (300, 300))
			imdataset.append(imdata_resized)
	sift = cv2.xfeatures2d.SIFT_create()
	alg = cv2.KAZE_create()

	feature_dataset = []
	for x in range(len(imdataset)):
		train_image = imdataset[x]
		getfeatures = extract_features(train_image, alg)
		#getfeatures = getFeatureMap(train_image,sift)
		feature_dataset.append(getfeatures)

	return(feature_dataset,alg,imdataset)



def forming_shadow(feature_dataset,alg,imdataset):
	kmin = 5
	final_image_created = np.zeros(shape=[300, 300])

	test_images = []
	for root, dirnames, filenames in os.walk("./dataset/test_images"):
		for filename in filenames:
			filepath = os.path.join(root, filename)
			test_image = ndimage.imread(filepath, mode="L")
			test_image_resized = misc.imresize(test_image, (300, 300))
			for temp in range(300):
				for temp2 in range(300):
					test_image_resized[temp][temp2] = 255 - test_image_resized[temp][temp2]  
			test_images.append(test_image_resized)

	for a in range(len(test_images)):
		#print(a)
		#print(len(test_images))

		total_match_value = 0;
		new_image_formed = np.zeros(shape=[300, 300])
		
		get_test_image = test_images[a]
		test_feature = extract_features(get_test_image, alg)
		#test_feature = getFeatureMap(get_test_image,sift)
		Match_value = np.zeros(len(imdataset))
		for z in range(len(imdataset)):
			getMatchPair = feature_dataset[z]
			getMatchValue = getMatch(test_feature,getMatchPair)
			Match_value[z] = getMatchValue
		idx = np.argsort(Match_value)

		for im in range(kmin):
			for x_image in range(300):
				for y_image in range(300):
					if(imdataset[idx[im]][x_image][y_image] < 200):
						#new_image_formed[x_image][y_image] = new_image_formed[x_image][y_image] + 1 - Match_value[idx[im]]
						new_image_formed[x_image][y_image] = new_image_formed[x_image][y_image] + ((2.31)**(-1*Match_value[idx[im]]))

		for ind in range(kmin):
			#total_match_value = total_match_value + 1 - Match_value[idx[ind]]
			total_match_value = total_match_value + (2.31**(-1*Match_value[idx[ind]]))

		for x_image in range(300):
				for y_image in range(300):
					final_image_created[x_image][y_image] = int((1.0 - (float(new_image_formed[x_image][y_image])/float(total_match_value)) )*255.0);
					#if(final_image_created[x_image][y_image] > 120 and final_image_created[x_image][y_image] < 200 ):
						#print(final_image_created[x_image][y_image])
		
		blur = cv2.blur(final_image_created,(15,15))
		for x_image in range(300):
				for y_image in range(300):
					if(get_test_image[x_image][y_image] < 200):
						blur[x_image][y_image] = 150
						#final_image_created[x_image][y_image] = 0;
						#print(get_test_image[x_image][y_image])
		#plt.imshow(blur,cmap = 'gray');
		plt.figure(a)
		plt.imshow(blur,cmap = 'gray')
		#plt.subplot(121),plt.imshow(get_test_image,cmap = 'gray')
		#plt.subplot(122),plt.imshow(blur,cmap = 'gray')
		plt.show()
		plt.pause(4)
		plt.close()
	
'''
		print(Match_value[idx[:kmin]])
		
		plt.figure(a)
		plt.subplot(231),plt.imshow(get_test_image,cmap = 'gray')
		plt.subplot(232),plt.imshow(imdataset[idx[0]],cmap = 'gray')
		plt.subplot(233),plt.imshow(imdataset[idx[1]],cmap = 'gray')
		plt.subplot(234),plt.imshow(imdataset[idx[2]],cmap = 'gray')
		plt.subplot(235),plt.imshow(imdataset[idx[3]],cmap = 'gray')
		plt.subplot(236),plt.imshow(imdataset[idx[4]],cmap = 'gray')
	plt.show()
'''
	
def main():
	"""Main function."""
	feature_dataset,alg,imdataset = make_bag_of_words()
	forming_shadow(feature_dataset,alg,imdataset)
	
if __name__=='__main__':
	main()
	
	



	

	

"""

#[NoOfImages, length, breadth] = [len(images), len(images[0]), len(images[0][0])]

#getting the edges 
edges = []
for i in range(NoOfImages):
	getImage = images[i]
	getedges = cv2.Canny(getImage,100,200)
	edges.append(getedges)


[NoOfEdges, lenEdges, brEdges] = [len(edges), len(edges[0]), len(edges[0][0])]
#print(NoOfEdges, lenEdges, brEdges)

first_image = edges[0]
second_image = edges[1]
third_image = edges[2]

sift = cv2.xfeatures2d.SIFT_create()
feature_map1 = getFeatureMap(first_image)
feature_map2 = getFeatureMap(second_image)
feature_map3 = getFeatureMap(third_image)

match_value1 = getMatch(feature_map1,feature_map2)
match_value2 = getMatch(feature_map2,feature_map3)
match_value3 = getMatch(feature_map3,feature_map1)

print(match_value1,match_value2, match_value3)
cv2.imshow('Matched Features', first_image)
cv2.waitKey(0)
cv2.destroyWindow('Matched Features')

"""







#kp1, des1 = sift.detectAndCompute(first_image,None)
#kp2, des2 = sift.detectAndCompute(second_image,None)

#bf = cv2.BFMatcher()
#matches = bf.match(des1,des2)
#matches = sorted(matches, key=lambda val: val.distance)
#img3 = drawMatches(first_image,kp1,second_image,kp2,matches[:50])

# Show the image
#cv2.imshow('Matched Features', img3)
#cv2.waitKey(0)
#cv2.destroyWindow('Matched Features')


