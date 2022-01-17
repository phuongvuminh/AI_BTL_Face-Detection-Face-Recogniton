import os,glob
import numpy as np
import cv2, time
import mxnet as mx
import face_detection, face_model
from scipy.spatial.distance import euclidean, cosine
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

model_path = os.path.join(os.path.dirname(__file__), 'models/model-r100-ii/model')                  #lay duong dan cua model
model_recognition = face_model.FaceModel(mx.cpu(0), model_path, 0)									# dua model vao recogniton
retina_detector = face_detection.get_model('retinaface_r50_v1')										# goi module face_detection.py va dua model vao detection
retina_detector.prepare(mx.cpu(0))

def detecter(path_img,face,id):

	image = cv2.imread(path_img,cv2.IMREAD_COLOR)
	if image is None:
		return 0
	image_detect = image.copy()
	facesbbox, landmarks = retina_detector.detect(image_detect, threshold=0.5)
	
	for ( rectangle, point_mtcnn) in zip (facesbbox, landmarks):                #in cac cap gia tri tuong ung trong (facesbbox, landmarks))
		rectangle_bbox =np.array([rectangle])									# mang cac rectangle
		landmask_pst5 = point_mtcnn												# cac diem landmark tren khuon mat
		img_align = face_detection.face_alignment(image_detect, rectangle_bbox, landmask_pst5)		# align khuon mat
		cv2.imwrite("E:/AI/BTL/Work_AI_BTL/Face_Recognition/photo/Face/phuong/%s_%s.jpg"%(face,id), img_align)   #ghi khuon mat da align ra anh

def detect_align():
    array_filepath = []
    for subdir, dirs, files in os.walk("E:/AI/BTL/Work_AI_BTL/Face_Recognition/Image"):
        filepath = subdir + os.sep
        array_filepath.append(filepath)
    
    array_img = []
    for i in range(len(array_filepath)):
        files = glob.glob(array_filepath[i]+"*.jpg")
        array_img = array_img + files
    
    array_paths = []
    for img in array_img:
        array_paths.append(img.split('\\')[1])
    for i in range(len(array_img)):
        detecter(array_img[i],array_paths[i],i+1)

if __name__ == '__main__':
    # detecter('E:/AI/BTL/Work_AI_BTL/Face_Recognition/Image\\hung\\3_10.jpg',"hung",1)
    detect_align()
		