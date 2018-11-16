import cv2
import os
import math
from scipy import ndimage, misc
import numpy as np
from matplotlib import pyplot as plt
from scipy import spatial
from collections import defaultdict


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
			#print(dsc)
		else:
			return(np.zeros(vector_size * 64))
	except cv2.error as e:
		print 'Error: ', e
		return None
	idx = dsc.argsort()[-1:-411:-1]
	binarize_dsc = np.zeros(2048)
	binarize_dsc[idx[:]] = 1
	return(binarize_dsc)
	#return(dsc)


def allowPermute(feature_dataset,get_permute):
	new_feature_dataset = np.zeros(2048)
	ans = 2049
	for a in range(len(get_permute)):
		if(feature_dataset[get_permute[a]] != 0.0):
			ans = a
			break
	return(ans)



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
        test_image_resized = misc.imresize(test_image, (480, 480))
        test_images.append(test_image_resized)



imdataset = []
for root, dirnames, filenames in os.walk("./dataset/image_dataset"):
    for filename in filenames:
        filepath = os.path.join(root, filename)
        imdata = ndimage.imread(filepath, mode="L")
        imdata_resized = misc.imresize(imdata, (300, 300))
        imdataset.append(imdata_resized)

edges = []
for i in range(len(test_images)):
	getImage = test_images[i]
	getedges = cv2.Canny(getImage,100,200)
	edges.append(getedges)

train_edges = []
for i in range(len(imdataset)):
	getImage = imdataset[i]
	getedges = cv2.Canny(getImage,100,200)
	train_edges.append(getedges)

test_images = edges
imdataset = train_edges

trainImagePatch = []
for n in range(len(imdataset)):
	getEdgeImage = imdataset[n]
	gettrainImagePatch = []
	for x in range(9):
		for y in range(9):
			image_patch = getEdgeImage[x*30:x*30+60:1, y*30:y*30+60:1]
			gettrainImagePatch.append(image_patch)
	trainImagePatch.append(gettrainImagePatch)

testImagePatch = []
for n in range(len(test_images)):
	getEdgeImage = test_images[n]
	gettestImagePatch = []
	for x in range(18):
		for y in range(18):
			image_patch = getEdgeImage[x*24:x*24+96:1, y*24:y*24+96:1]
			gettestImagePatch.append(image_patch)
	testImagePatch.append(gettestImagePatch)	

train_feature_dataset = []
for x in range(len(trainImagePatch)):
	feature_dataset = []
	for y in range(len(trainImagePatch[0])):
		train_image = trainImagePatch[x][y]
		getfeatures = extract_features(train_image)
		feature_dataset.append(getfeatures)
	train_feature_dataset.append(feature_dataset)

test_feature_dataset = []
for a in range(len(testImagePatch)):
	feature_dataset = []
	for b in range(len(testImagePatch[0])):
		get_test_image = testImagePatch[a][b]
		test_feature = extract_features(get_test_image)
		feature_dataset.append(test_feature)
	test_feature_dataset.append(feature_dataset)


numOfPermute = 20
kOfPermute = 3
#print(len(arrayIndex),len(arrayIndex[0]))
testarrayIndex = np.zeros([numOfPermute,len(test_feature_dataset),len(test_feature_dataset[0])])
power_array = np.zeros(3)
getMatchIndex = defaultdict(list)
power_array[0] = 1
power_array[1] = 10
power_array[2] = 100

for numpermute in range(numOfPermute):
	arrayIndex = np.zeros([len(train_feature_dataset),len(train_feature_dataset[0])])
	for p in range(kOfPermute):
		permute_value = np.random.permutation(2048)
		
		for x in range(len(train_feature_dataset)):
			for z in range(len(train_feature_dataset[0])):
				getindex = allowPermute(train_feature_dataset[x][z],permute_value)
				arrayIndex[x][z] = (getindex*power_array[p]) + arrayIndex[x][z]
			#print(arrayIndex[x])
		
		for y in range(len(test_feature_dataset)):
			for w in range(len(test_feature_dataset[0])):
				getindex2 = allowPermute(test_feature_dataset[y][w],permute_value)
				testarrayIndex[numpermute][y][w] = (getindex2*power_array[p]) + testarrayIndex[numpermute][y][w]
			#print(testarrayIndex[numpermute][y])

		#print([arrayIndex[0],testarrayIndex[numpermute][0]])
	for g in range(len(train_feature_dataset)):
		for h in range(len(train_feature_dataset[0])):
			getMatchIndex[arrayIndex[g][h]].append(g)

kmin = 6
for m in range(len(test_feature_dataset)):
	Match_value = np.zeros(len(imdataset))
	for n in range(len(test_feature_dataset[0])):
		get_test_image = testImagePatch[m][n]
		for num in range(numOfPermute):
			for j in range(len(getMatchIndex[testarrayIndex[num][m][n]])):
				Match_value[getMatchIndex[testarrayIndex[num][m][n]][j]] = Match_value[getMatchIndex[testarrayIndex[num][m][n]][j]] + 1
	idxMostmatched = Match_value.argsort()[-1:-kmin:-1]
	plt.figure(m)
	plt.subplot(231),plt.imshow(test_images[m],cmap = 'gray')
	plt.subplot(232),plt.imshow(imdataset[idxMostmatched[0]],cmap = 'gray')
	plt.subplot(233),plt.imshow(imdataset[idxMostmatched[1]],cmap = 'gray')
	plt.subplot(234),plt.imshow(imdataset[idxMostmatched[2]],cmap = 'gray')
	plt.subplot(235),plt.imshow(imdataset[idxMostmatched[3]],cmap = 'gray')
	plt.subplot(236),plt.imshow(imdataset[idxMostmatched[4]],cmap = 'gray')

plt.show()
"""
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
"""
