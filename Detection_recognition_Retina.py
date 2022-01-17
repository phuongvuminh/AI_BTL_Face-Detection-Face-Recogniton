import sys, os,glob
import numpy as np
import cv2, time, json
import uuid, io
import mxnet as mx
import face_detection, face_model
import csv, math
from scipy.spatial.distance import euclidean, cosine
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

model_path = os.path.join(os.path.dirname(__file__), 'models/model-r100-ii/model')                  #lay duong dan cua model
model_recognition = face_model.FaceModel(mx.cpu(0), model_path, 0)									# dua model vao recogniton
retina_detector = face_detection.get_model('retinaface_r50_v1')										# goi module face_detection.py va dua model vao detection
retina_detector.prepare(mx.cpu(0))


#detect ket hop recognition
def detecter_recognition(path_img):

	image = cv2.imread(path_img)
	if image is None:
		return 0
	image_detect = image.copy()                    								#copy anh doc vao sang anh de detect
	facesbbox, landmarks = retina_detector.detect(image_detect, threshold=0.5)
	feat =[]
	for ( rectangle, point_mtcnn) in zip (facesbbox, landmarks):
		rectangle_bbox =np.array([rectangle])
		landmask_pst5 = point_mtcnn
		img_align = face_detection.face_alignment(image_detect, rectangle_bbox, landmask_pst5)
		feature_face = model_recognition.get_feature(img_align)
		feat = feature_face.tolist()
		break
	return feat

#detect truoc roi dua anh vao recogn
def detecter(path_img):

	image = cv2.imread(path_img)
	if image is None:
		return 0
	image_detect = image.copy()
	facesbbox, landmarks = retina_detector.detect(image_detect, threshold=0.5)
	
	for ( rectangle, point_mtcnn) in zip (facesbbox, landmarks):                #in cac cap gia tri tuong ung trong (facesbbox, landmarks))
		face_id = str(uuid.uuid4().hex)											# voi moi khuon mat gan mot id rieng su dung uuid4
		rectangle_bbox =np.array([rectangle])									# mang cac rectangle
		landmask_pst5 = point_mtcnn												# cac diem landmark tren khuon mat
		img_align = face_detection.face_alignment(image_detect, rectangle_bbox, landmask_pst5)		# align khuon mat
		cv2.imwrite("/media/nguyenkhactuananh/01D447AEA1BA74E0/Work_AI/Face_Recognition/Image/Face_align_%s.jpg"%face_id, img_align)   #ghi khuon mat da align ra anh
		
def check_detector(path_img):
	image = cv2.imread(path_img)
	if image is None:
		return 0
	image_detect = image.copy()
	start_time = time.time()
	facesbbox, landmarks = retina_detector.detect(image_detect, threshold=0.5)	
	print("Time to use algorithm: %s" %(time.time()-start_time))
	if facesbbox is not None:
		print("find total face: ", facesbbox.shape[0],' faces')
		for i in range(facesbbox.shape[0]):
			box = facesbbox[i].astype(np.int)
			color = (0,0,255)
			cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),color,2)
			font = cv2.FONT_HERSHEY_SIMPLEX
			blur= facesbbox[i,4]
			k = "%.4f"%blur
			cv2.putText(image,k,(box[0]+2,box[1]+12),font,0.6,(0,255,0),2)

			if landmarks is not None:
				landmask_pst5 = landmarks[i].astype(np.int)
				for l in range(landmask_pst5.shape[0]):
					color=(0,0,255)
					if l==0 or l==3:
						color = (0,255,0)
					cv2.circle(image, (landmask_pst5[l][0],landmask_pst5[l][1]),1,color,2)
	filename = 'E:/AI/BTL/Work_AI_BTL/Face_Recognition/detect_test.jpg'
	print('writing', filename)
	cv2.imwrite(filename, image)


def recognition(path_img):
	image_Bbface = cv2.imread(path_img)
	feature_face = model_recognition.get_feature(image_Bbface)   #dua anh vao model va trich xuat dac trung cua anh
	feat = feature_face.tolist()							 
	return feat                                                  # ket qua tra ve la mot mang
	
	
def loadData(path):
    f = open(path, "r")
    data = csv.reader(f) #csv format
    data = np.array(list(data))# covert to matrix
    np.random.shuffle(data) # shuffle data
    f.close()
    trainSet = data[:1688] #training data from 
    testSet = data[1688:]# the others is testing data
    return trainSet, testSet

def calcDistancs(pointA, pointB, numOfFeature=512):
    tmp = 0
    for i in range(numOfFeature):
        tmp += (float(pointA[i]) - float(pointB[i])) ** 2
    return math.sqrt(tmp)


def kNearestNeighbor(trainSet, point, k):
    distances = []
    for item in trainSet:
        distances.append({
            "label": item[-1],
            "value": calcDistancs(item, point)
        })
    distances.sort(key=lambda x: x["value"])
    labels = [item["label"] for item in distances]
    return labels[:k]


def findMostOccur(arr):
    labels = set(arr) # set label
    ans = ""
    maxOccur = 0
    for label in labels:
        num = arr.count(label)
        if num > maxOccur:
            maxOccur = num
            ans = label
    return ans


def similarity(pointA, pointB, numOfFeature=512):
	temp_similarity = 0
	for i in range(numOfFeature):
		temp_similarity += float(pointA[i]) * float(pointB[i])
	return abs(1-temp_similarity)                #abs(1- np.dot(feature_face1, feature_face2))



def main():
	dem = 0
	trainSet, testSet = loadData("E:/AI/BTL/Work_AI_BTL/Face_Recognition/feature_face1.csv")
	numOfRightAnwser = 0
	image = cv2.imread("E:/AI/BTL/Work_AI_BTL/Face_Recognition/Image/test1.jpg",cv2.IMREAD_COLOR)
	feature_face = detecter_recognition("E:/AI/BTL/Work_AI_BTL/Face_Recognition/Image/test1.jpg")
	for feature_face1 in trainSet:
		temp_similarity = similarity(feature_face,feature_face1)
		if temp_similarity <= 0.6:
			dem += 1
	
	if dem == 0:
			answer = "unknown" 
	else:
		knn = kNearestNeighbor(trainSet,feature_face,5)
		answer = findMostOccur(knn)
	


	""" ghi ra anh """
	image_detect = image.copy()                                                     #copy anh doc vao sang anh de detect
	facesbbox, landmarks = retina_detector.detect(image_detect, threshold=0.5)


	if facesbbox is not None:
		for i in range(facesbbox.shape[0]):
			box = facesbbox[i].astype(np.int)
			color = (0,0,255)
			cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),color,2)
			font = cv2.FONT_HERSHEY_SIMPLEX
			blur= facesbbox[i,4]
			cv2.putText(image,answer,(box[0]+2,box[1]+12),font,0.6,(0,255,0),2)


			if landmarks is not None:
				landmask_pst5 = landmarks[i].astype(np.int)
				for l in range(landmask_pst5.shape[0]):
					color=(0,0,255)
					if l==0 or l==3:
						color = (0,255,0)
					cv2.circle(image, (landmask_pst5[l][0],landmask_pst5[l][1]),1,color,2)
		    
	filename = 'E:/AI/BTL/Work_AI_BTL/Face_Recognition/recogn_test2.jpg'
	print('writing', filename)
	cv2.imwrite(filename, image)
	"""tinh accuracy """
	for item in testSet:
		knn = kNearestNeighbor(trainSet, item, 5)
		answer = findMostOccur(knn)
		numOfRightAnwser += item[-1] == answer
		#print("label: {} -> predicted: {}".format(item[-1], answer))
	
	print("Accuracy", numOfRightAnwser/len(testSet))

	
if __name__ == '__main__':
	main()
	

	
	


			
				
				
