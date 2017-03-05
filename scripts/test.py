import sys
sys.path.append("..")
from recognizer.predicter import Predicter
pre = Predicter()
print (pre.predict('../images/yazawa_nico/1.jpg'))
