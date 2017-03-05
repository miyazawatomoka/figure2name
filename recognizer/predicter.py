import sys
from PIL import Image
import numpy as np
from sklearn.externals import joblib
from recognizer.svm_classify import SvmClassify
sys.path.append("..")
import i2v
class Predicter(object):
    def __init__(self):
        self.svm = SvmClassify(clf_path = "../models/recognizer_svm_classify.pkl")
        self.illust2vec = i2v.make_i2v_with_caffe(
            "../models/illust2vec.prototxt", "../models/illust2vec_ver200.caffemodel")

    def classify(self, img):
        illust2vec = self.illust2vec; svm = self.svm;
        feature_vec = illust2vec.extract_feature([img])[0]
        class_result = svm.predict(feature_vec)
        return class_result

    def predict(self, img_path):
        img = Image.open(img_path)
        return self.classify(img)
