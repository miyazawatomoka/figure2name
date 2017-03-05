# Figure2Name
这是一个动画人物识别模块
## 用法
1. 首先下载必要的模型与数据集
2. 安装依赖
3. 运行 `scripts/tarin.py`训练模型
4. 如下进行人物识别
```python
from recognizer.predicter import Predicter
pre = Predicter()
name = pre.predict('../images/yazawa_nico/1.jpg')
print(name)
```
## 实现
### 思路
首先想到的实现思路就是利用人脸比对.但由于不同画师的画风不同等原因,动画人物的识别和传统的人脸比对也会略有不同.  
传统的人脸比对项目通过提取肤色矩阵-->图像灰度化-->提取人脸特征矩阵-->计算欧式距离完成.但是,由于作画的原因,导致了传统人脸识别中肤色矩阵与人脸特征矩阵不能有效.所以,传统的人脸识别方案并不适用.  
这里参考了人们识别动画人物时往往通过一些特征以识别,比如人物的眼型 发色等等.所以, 考虑对如上特征进行提取, 并进行分类.  
很自然的考虑是利用CNN, CNN可以对局部信息进行有效的提取.这里使用了illustration2tag pre-train model.它实现了对动画图片的tag提取,作者同时给出了一个生成一个4096维向量的model供使用.  
之后,通过svm对特征向量进行分类.最后达到了只需要使用15张图的训练,即可对一个人物进行识别.
### 依赖
* numpy
* caffe(请编译并安装caffe-python模块)
* pillow
* sklearn

## 资源

## 意见与建议
如果你有更好的实现思路，欢迎在issue中指出，或者给我发邮件honoka@honoka.cc
