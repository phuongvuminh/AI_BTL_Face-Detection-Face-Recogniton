from retina_detection import *
import face_align

_models = {
    'retinaface_r50_v1': retinaface_r50_v1,
    'retinaface_mnet025_v1': retinaface_mnet025_v1,
    'retinaface_mnet025_v2': retinaface_mnet025_v2,
}

def get_model(name, **kwargs):
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net

def face_alignment(image, bbox, pts5):
    if bbox.shape[0]==0:
        return None
    bbox = bbox[0, 0:3]
    pts5= np.array([pts5])
    pts5 = pts5[0, :]
    nimg = face_align.norm_crop(image, pts5)
    return nimg


# def Show_video():
# 	model_path = os.path.join(os.path.dirname(__file__), 'models/model-r100-ii/model')
# 	model = face_model.FaceModel(mx.gpu(0), model_path, 0)
# 	retina_detector = face_detection.get_model('retinaface_mnet025_v2')
# 	retina_detector.prepare(mx.gpu(0))
# 	img = cv2.imread('/home/muoi/Work/Counting_recognition/add_datamongo/Gallery/0_Son.png')
# 	facesbbox, landmarks = retina_detector.detect(img, threshold=0.8)
# 	for ( bbox, landmarks) in zip (facesbbox, landmarks):
# 		cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(255,0,0),2)
# 	cv2.imshow("xxxx", img)
# 	cv2.waitKey(0)