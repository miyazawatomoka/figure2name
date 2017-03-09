import numpy as np
from sklearn import svm
from collections import OrderedDict
import os
import code
from sklearn.externals import joblib
PATH = os.path.dirname(__file__)
class SvmClassify(object):
    def __init__(self, clf_path=""):
        nameList, x, y = self.__get_x_y()
        self.nameList = nameList; self.x = x; self.y = y;
        clf = svm.SVC(decision_function_shape='ovr', probability=True,  kernel='linear')
        self.clf = clf
        if (clf !=""):
            self.clf = joblib.load(clf_path)

    def train(self, path=os.path.join(PATH, "../models/recognizer_svm_classify.pkl")):
        self.clf.fit(self.x, self.y)
        joblib.dump(self.clf, path)

    def predict(self, X):
        clf = self.clf
        result = self.nameList[clf.predict([X])[0]]
        # proba_array = clf.predict_proba([feature])
        return result

    def __get_x_y(self):
        x = []
        y = []
        rootDir = os.path.join(PATH, '../images/')
        listDirs = os.walk(rootDir)
        nameList  = os.listdir(rootDir)  # 这里可用一个临时hash重构
        for root, dirs, files in listDirs:
            for f in list(filter(lambda x : x.endswith('.npy'), files)):
                path     =  os.path.join(root, f)
                name     =  os.path.split(root)[1]
                feature  =  np.load(path)
                x.append(feature)
                y.append(nameList.index(name))
        return nameList, x, y

    def test(self):
        rootDir  = os.path.join(PATH, '../test_images')
        nameList = self.nameList
        listDirs = os.walk(rootDir)
        clf        =  self.clf
        errorCount =  0
        testCount  =  0
        for root, dirs, files in listDirs:
            for f in list(filter(lambda x : x.endswith('.npy'), files)):
                testCount  += 1
                path       =  os.path.join(root, f)
                name       =  os.path.split(root)[1]
                feature    =  np.load(path)
                nameNumber =  nameList.index(name)
                predict_y  =  clf.predict([feature])[0]
                print(clf.predict_proba([feature]))
                if (predict_y != nameNumber):
                    print ("error in: %s" %path)
                    print("predict is %s" %nameList[predict_y])
                    errorCount += 1
                    print(clf.predict_proba([feature]))
        errorRate = errorCount / testCount
        # code.interact(local=dict(globals(), **locals()))
        print ("Accuracy rate is %f" %(1 - errorRate))
