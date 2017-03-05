import sys
sys.path.append("..")
from recognizer import FeatureExtracter, SvmClassify

extracter = FeatureExtracter()
extracter.extract()
classify = SvmClassify()
classify.train()
classify.test()
