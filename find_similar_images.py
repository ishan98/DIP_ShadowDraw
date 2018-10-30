import cv2
import os
import math
from scipy import ndimage, misc
import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial

def extract_features(image, vector_size=32):
	#print('Hello')
	try:
		alg = cv2.KAZE_create()
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
			#print(len(dsc))
		else:
			return(np.zeros(vector_size * 64))
	except cv2.error as e:
		print 'Error: ', e
		return None
	return(dsc)




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


#resizing the images
test_images = []
for root, dirnames, filenames in os.walk("./dataset/test_image"):
    for filename in filenames:
        filepath = os.path.join(root, filename)
        test_image = ndimage.imread(filepath, mode="L")
        test_image_resized = misc.imresize(test_image, (300, 300))
        test_images.append(test_image_resized)

imdataset = []
for root, dirnames, filenames in os.walk("./dataset/image_dataset"):
    for filename in filenames:
        filepath = os.path.join(root, filename)
        imdata = ndimage.imread(filepath, mode="L")
        imdata_resized = misc.imresize(imdata, (300, 300))
        imdataset.append(imdata_resized)

feature_dataset = []
for x in range(len(imdataset)):
	train_image = imdataset[x]
	getfeatures = extract_features(train_image)
	feature_dataset.append(getfeatures)

kmin = 5
for a in range(len(test_images)):
	get_test_image = test_images[a]
	test_feature = extract_features(get_test_image)
	Match_value = np.zeros(len(imdataset))
	for z in range(len(imdataset)):
		getMatchPair = feature_dataset[z]
		getMatchValue = getMatch(test_feature,getMatchPair)
		Match_value[z] = getMatchValue
	idx = np.argsort(Match_value)
	print(Match_value[idx[:kmin]])
	plt.figure(a)
	plt.subplot(231),plt.imshow(get_test_image,cmap = 'gray')
	plt.subplot(232),plt.imshow(imdataset[idx[0]],cmap = 'gray')
	plt.subplot(233),plt.imshow(imdataset[idx[1]],cmap = 'gray')
	plt.subplot(234),plt.imshow(imdataset[idx[2]],cmap = 'gray')
	plt.subplot(235),plt.imshow(imdataset[idx[3]],cmap = 'gray')
	plt.subplot(236),plt.imshow(imdataset[idx[4]],cmap = 'gray')
plt.show()

