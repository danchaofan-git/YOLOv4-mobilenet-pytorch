# YOLOv4-mobilenet-pytorch
# 本项目代码参考作者bubbliiiing，原项目地址为：https://github.com/bubbliiiing/mobilenet-yolov4-lite-pytorch
## 目录
1. [性能情况 Performance](#性能情况)
2. [所需环境 Environment](#所需环境)
3. [注意事项 Attention](#注意事项)
4. [小技巧的设置 TricksSet](#小技巧的设置)
5. [文件下载 Download](#文件下载)
6. [预测步骤 How2predict](#预测步骤)
7. [训练步骤 How2train](#训练步骤)
8. [各脚本功能](#各脚本功能)
9. [参考资料 Reference](#Reference)

## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| VOC07+12 | [yolov4_mobilenet_v1_voc.pth]| VOC-Test07 | 416x416 | - | 79.72
| VOC07+12 | [yolov4_mobilenet_v2_voc.pth] | VOC-Test07 | 416x416 | - | 80.12
| VOC07+12 | [yolov4_mobilenet_v3_voc.pth] | VOC-Test07 | 416x416 | - | 79.01

## 所需环境
torch==1.7.0

## 注意事项
提供的三个权重分别是基于mobilenetv1、mobilenetv2、mobilenetv3主干网络训练而成的。使用的时候注意backbone和权重的对应。   
训练前注意修改model_path和backbone使得二者对应。   
预测前注意修改model_path和backbone使得二者对应。   

## 小技巧的设置
在train.py文件下：   
1、mosaic参数可用于控制是否实现Mosaic数据增强。    
2、Cosine_scheduler可用于控制是否使用学习率余弦退火衰减。    
3、label_smoothing可用于控制是否Label Smoothing平滑。   

## 文件下载 
训练所需的各个权值可在百度网盘中下载。    
链接: https://pan.baidu.com/s/1mX5UPqPwGz-A1son3vWxqw 提取码: 1kut     
三个已经训练好的权重均为VOC数据集的权重。  
  
## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，在百度网盘下载权重，放入model_data，运行predict.py，输入  
```python
img/street.jpg
``` 
2. 利用video.py可进行摄像头检测。  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    "model_path"        : 'model_data/yolov4_mobilenet_v2_map76.93.pth',
    "anchors_path"      : 'model_data/yolo_anchors.txt',
    "classes_path"      : 'model_data/voc_classes.txt',
    "backbone"          : 'mobilenetv2',
    
    "model_image_size"  : (416, 416, 3),
    "confidence"        : 0.5,
    "iou"               : 0.3,
    "cuda"              : True
}
```
3. 运行predict.py，输入  
```python
img/street.jpg
```
4. 利用video.py可进行摄像头检测。  

## 训练步骤
1. 本文使用VOC格式进行训练。  
2. 训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
3. 训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages下的train文件夹中。  
4. 在训练前利用voc2yolo4.py文件生成对应的txt。  
5. 再运行根目录下的voc_annotation.py，运行前需要将classes改成自己的classes。**注意不要使用中文标签，文件夹中不要有空格！**   
```python
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
```
6. 此时会生成对应的2007_train.txt，每一行对应其**图片位置**及其**真实框的位置**。  
7. **在训练前需要务必在model_data下新建一个txt文档，文档中输入需要分的类，在train.py中将classes_path指向该文件**，示例如下：   
```python
classes_path = 'model_data/new_classes.txt'    
```
model_data/new_classes.txt文件内容为：   
```python
cat
dog
...
```
8. 运行train.py即可开始训练。

## 各脚本功能
1. train.py---训练脚本
2. test.py---测试网络结构
3. predict.py---测试脚本
4. DataAugmentForObject.py---数据增强脚本，用来扩充数据集
5. crop_boudingbox.py---将打标好的xml文件里包含的真实框切分，保存下来，可用作图片分类
6. get_results.py---批量测mAP等指标，会遍历logs文件夹里的权重文件，然后一一进行测试
7. video.py---视频检测脚本
8. FPS_test.py---测试FPS
9. ui.py---用Pyside6写的UI界面
10. kmeans_for_anchor.py---kmeans聚类选出先验框
11. utils文件夹中的demo.py可以可视化kmeans

## Reference
https://github.com/qqwweee/keras-yolo3/  
https://github.com/Cartucho/mAP  
https://github.com/Ma-Dan/keras-yolo4 
