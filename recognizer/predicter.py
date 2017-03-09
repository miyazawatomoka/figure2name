import sys, os
from PIL import Image
import numpy as np
from sklearn.externals import joblib
from recognizer.svm_classify import SvmClassify
sys.path.append("..")
import i2v
PATH = os.path.dirname(__file__)

class Predicter(object):
    def __init__(self):
        clf_path = os.path.join(PATH, "../models/recognizer_svm_classify.pkl")
        proto_path = os.path.join(PATH, "../models/illust2vec.prototxt")
        model_path = os.path.join(PATH, "../models/illust2vec_ver200.caffemodel")
        self.svm = SvmClassify(clf_path = clf_path)
        self.illust2vec = i2v.make_i2v_with_caffe(
            proto_path, model_path)

    def classify(self, img):
        illust2vec = self.illust2vec; svm = self.svm;
        feature_vec = illust2vec.extract_feature([img])[0]
        class_result = svm.predict(feature_vec)
        return class_result

    def predict(self, img_path):
        img = Image.open(img_path)
        return self.classify(img)
