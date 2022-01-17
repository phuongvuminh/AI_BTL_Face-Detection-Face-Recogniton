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

def getFeature(path_img):

	image = cv2.imread(path_img)
	if image is None:
		return 0
	image_detect = image.copy()  
	feature_face = model_recognition.get_feature(image_detect)   
	feat = feature_face.tolist()                							
	return feat

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

def main():

	array_filepath = []
    
	for subdir, dirs, files in os.walk("E:/AI/BTL/Work_AI_BTL/Face_Recognition/AFDB"):
		filepath = subdir + os.sep
		array_filepath.append(filepath)

	array_img = []
	for i in range(len(array_filepath)):
		files = glob.glob(array_filepath[i]+"*.jpg")
		array_img = array_img + files
    
	array_paths = []
	for img in array_img:
		array_paths.append(img.split('\\')[1])
	#print("img: ", len(array_img))

	"""trich xuat vecto dac trung va label cua anh"""
	f = open("E:/AI/BTL/Work_AI_BTL/Face_Recognition/feature_face1.csv", "a", newline="")
	writer = csv.writer(f)
	for i in range(len(array_img)):
		feature_face=detecter_recognition(array_img[i])
		feature_face.append(array_paths[i])
		writer.writerow(feature_face)
	f.close
	

if __name__ == '__main__':
	main()