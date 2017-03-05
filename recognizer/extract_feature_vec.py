import sys
from PIL import Image
import code, os
import numpy as np
sys.path.append("..")
import i2v
class FeatureExtracter(object):
    def __getImgs(self, rootDir):
        imgs  = []
        paths = []
        list_dirs = os.walk(rootDir)
        for root, dirs, files in list_dirs:
            for f in list(filter(lambda x : not x.endswith('.npy'), files)):
                path = os.path.join(root, f)
                if (os.path.exists(os.path.splitext(path)[0]+'.npy')):
                    continue
                imgs.append(Image.open(path))
                paths.append(path)
        return imgs, paths


    def extractFeatureVec(self, rootDir):
        imgs, paths = self.__getImgs(rootDir)
        if not imgs:
            return
        illust2vec = i2v.make_i2v_with_caffe(
            "../models/illust2vec.prototxt", "../models/illust2vec_ver200.caffemodel")
        results = illust2vec.extract_feature(imgs)
        for result in results:
            path = os.path.splitext(paths.pop(0))[0]
            np.save(path, result)

    def extract(self):
        self.extractFeatureVec('../images/')
        self.extractFeatureVec('../test_images/')
        print("extract succ")
