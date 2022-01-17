import os
import numpy as np
import cv2
import mxnet as mx
import face_detection, face_model
from scipy.spatial.distance import euclidean, cosine
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

model_path = os.path.join(os.path.dirname(__file__), 'models/model-r100-ii/model')                  #lay duong dan cua model
model_recognition = face_model.FaceModel(mx.cpu(0), model_path, 0)									# dua model vao recogniton
retina_detector = face_detection.get_model('retinaface_r50_v1')										# goi module face_detection.py va dua model vao detection
retina_detector.prepare(mx.cpu(0))

cap = cv2.VideoCapture(0)
sampleNum = 0

while(True):
    ret, image = cap.read()
    sampleNum += 1
    image_detect = image.copy()
    facesbbox, landmarks = retina_detector.detect(image_detect, threshold=0.5)
    for ( rectangle, point_mtcnn) in zip (facesbbox, landmarks):                #in cac cap gia tri tuong ung trong (facesbbox, landmarks))										
        rectangle_bbox =np.array([rectangle])									
        landmask_pst5 = point_mtcnn												
        img_align = face_detection.face_alignment(image_detect, rectangle_bbox, landmask_pst5)		
        cv2.imwrite('E:/AI/BTL/Work_AI_BTL/Face_Recognition/photo/Face/toan/1_'+str(sampleNum)+'.jpg', img_align)   

    cv2.imshow('frame', image)
    cv2.waitKey(1)

    if(sampleNum >50):
        break

cap.release()
cv2.destroyAllWindows()
# voi moi khuon mat gan mot id rieng su dung uuid4
