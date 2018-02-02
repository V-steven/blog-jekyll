---
layout: post
title: UAV monitoring based on DPM & KCF
tags:  [图像识别]
excerpt: "通过DPM和KCF算法来实现无人机的监测                                          "
---

**目录：**

**1.DPM和KCF融合;**

**2.DPM训练自己的模型(本博客的重点);**

**3.效果展示;**

----
----


### 1.1 首先初步用人形监测模型调试，效果如下

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/UAV.gif" style="width:500px">  

----
### 1.2 DPM和KCF共同完成监测

- (1)DPM是非常成功的目标检测算法`(具体了解查相关资料)`，用官方的人形模型识别，平均在0.8s左右处理一张图片，如果用在视频监测上的效率是非常低的`(一帧左右)`，所以采用KCF跟踪算法共同完成运动物体的识别监测。

- (2) 两算法的融合，无非就是进行相关参数的相互传递，当DPM对运动物体识别成功后，将坐标传递给KCF进行跟踪。运动物体消失在监测区域，则DPM恢复识别功能。具体结构如下图：

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/kcf_dpm.png" style="width:400px">  

---
---

### 2.1 训练环境

- (1) Ubuntu14.4+Matlab2012a+gcc4.4.7+voc-release5

---

### 2.2 编译源代码

- (1)下载voc-release5[[code]](https://github.com/rbgirshick/voc-dpm);下载 PASCAL VOC devkit 2011[[code]](http://pan.baidu.com/s/1i4T7IQD)。
- (2)将devkit放入voc-release5文件中的voc2007中。
- (3)打开matlab，在voc-release5文件下运行：

> addpath(genpath('.'))   
> compile   
> demo  
> cascade_demo  

---

### 2.3 训练相关步骤

- (1)准备训练样本，样本包括正样本和负样本。我是训练的六旋翼模型，所以找了400张正样本(含六旋翼的图片)，500张负样本(不含六旋翼的图片)。

- (2)样本图片处理，将图片统一处理成了520*390大小，分别保存到pos-new和neg-new文件下，python写好的图片批量处理：
{% highlight python %}
#coding:utf-8
import os
from PIL import Image

'''
shear image 520*390
gump 2016-03-29

'''

path = 'F:/DPM/d'
PATH = os.listdir(path)
i = 0

for img in PATH:
    format = img[img.find('.'):]
    if ( format == '.jpg' or format == '.png'):

        im = Image.open(img)
        xsize,ysize=im.size

        if(xsize >= 520 or ysize >= 390):
            i = i + 1
            box=(xsize / 2 - 260, ysize / 2- 195, xsize / 2 + 260, ysize / 2 + 195) #center
            result = im.crop(box)
            if(i < 10):
                result.save('F:/DPM/neg-new/00' + str(i) + '.jpg')
            elif(i >= 10 and i < 100):
                result.save('F:/DPM/neg-new/0' + str(i) + '.jpg')
            else:
                result.save('F:/DPM/neg-new/' + str(i) + '.jpg')
{% endhighlight %}
---

- (3) 读取数据集。修改voc-release5/data/pascal_data.m文件：

{% highlight c %}
function [pos, neg, impos] = pascal_data(cls, year)
% Get training data from my own dataset
%   [pos, neg, impos] = pascal_data(cls, year)
%
% Return values
%   pos     Each positive example on its own
%   neg     Each negative image on its own
%   impos   Each positive image with a list of foreground boxes


conf      = voc_config('pascal.year', year);
dataset_fg=conf.training.train_set_fg;
cachedir  =conf.paths.model_dir;

%added by yihanglou  using my own img and txtinfo
PosImageFile = '/home/gump/DOCU/voc-release5-New/UAV_data/IMG_list/posData.txt'; %正样本数据格式文件
NegImageFile = '/home/gump/DOCU/voc-release5-New/UAV_data/IMG_list/negData.txt'; %负样本数据格式文件
BasePath = '/home/gump/DOCU/voc-release5-New/UAV_data/IMG_train'; %正负样本图片文件

pos      = [];
impos    = [];
numpos   = 0;
numimpos = 0;
dataid   = 0;

fin = fopen(PosImageFile,'r');

now = 1;

while ~feof(fin)
    line = fgetl(fin);
    S = regexp(line,' ','split');
    count = str2num(S{2});
    fprintf('%s: parsing positives (%s): %d\n', ...
             cls, S{1}, now);
    now = now + 1;
    for i = 1:count;
        numpos = numpos + 1;
        dataid = dataid + 1;
        bbox = [str2num(S{i*4-1}),str2num(S{i*4}),str2num(S{i*4+1}),str2num(S{i*4+2})];

        pos(numpos).im      = [BasePath '/' S{1}];
        pos(numpos).x1      = bbox(1);
        pos(numpos).y1      = bbox(2);
        pos(numpos).x2      = bbox(3);
        pos(numpos).y2      = bbox(4);
        pos(numpos).boxes   = bbox;
        pos(numpos).flip    = false;
        pos(numpos).trunc   = 0;%1 represent incomplete objects, 0 is complete
        pos(numpos).dataids = dataid;
        pos(numpos).sizes   = (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1);

        img = imread([BasePath '/' S{1}]);
        [height, width, depth] = size(img);

        % Create flipped example
        numpos  = numpos + 1;
        dataid  = dataid + 1;
        oldx1   = bbox(1);
        oldx2   = bbox(3);
        bbox(1) = width - oldx2 + 1;
        bbox(3) = width - oldx1 + 1;

        pos(numpos).im      = [BasePath '/' S{1}];
        pos(numpos).x1      = bbox(1);
        pos(numpos).y1      = bbox(2);
        pos(numpos).x2      = bbox(3);
        pos(numpos).y2      = bbox(4);
        pos(numpos).boxes   = bbox;
        pos(numpos).flip    = true;
        pos(numpos).trunc   = 0;% to make operation simple
        pos(numpos).dataids = dataid;
        pos(numpos).sizes   = (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1);

    end

    % Create one entry per foreground image in the impos array
    numimpos                = numimpos + 1;
    impos(numimpos).im      = [BasePath '/' S{1}];
    impos(numimpos).boxes   = zeros(count, 4);
    impos(numimpos).dataids = zeros(count, 1);
    impos(numimpos).sizes   = zeros(count, 1);
    impos(numimpos).flip    = false;

    for j = 1:count
        dataid = dataid + 1;
        bbox   = [str2num(S{j*4-1}),str2num(S{j*4}),str2num(S{j*4+1}),str2num(S{j*4+2})];

        impos(numimpos).boxes(j,:) = bbox;
        impos(numimpos).dataids(j) = dataid;
        impos(numimpos).sizes(j)   = (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1);
    end     

    img = imread([BasePath '/' S{1}]);
    [height, width, depth] = size(img);

     % Create flipped example
    numimpos                = numimpos + 1;
    impos(numimpos).im      = [BasePath '/' S{1}];
    impos(numimpos).boxes   = zeros(count, 4);
    impos(numimpos).dataids = zeros(count, 1);
    impos(numimpos).sizes   = zeros(count, 1);
    impos(numimpos).flip    = true;
    unflipped_boxes         = impos(numimpos-1).boxes;


    for j = 1:count
    dataid  = dataid + 1;
    bbox    = unflipped_boxes(j,:);
    oldx1   = bbox(1);
    oldx2   = bbox(3);
    bbox(1) = width - oldx2 + 1;
    bbox(3) = width - oldx1 + 1;

    impos(numimpos).boxes(j,:) = bbox;
    impos(numimpos).dataids(j) = dataid;
    impos(numimpos).sizes(j)   = (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1);
    end
end

fclose(fin);
% Negative examples from the background dataset

fin2 = fopen(NegImageFile,'r');
neg    = [];
numneg = 0;
negnow = 0;
while ~feof(fin2)
     line = fgetl(fin2);
     fprintf('%s: parsing Negtives (%s): %d\n', ...
                   cls, line, negnow);

     negnow             = negnow +1;
     dataid             = dataid + 1;
     numneg             = numneg+1;
     neg(numneg).im     = [BasePath '/' line];
     disp(neg(numneg).im);
     neg(numneg).flip   = false;
     neg(numneg).dataid = dataid;
 end

 fclose(fin2);
 save([cachedir cls '_' dataset_fg '_' year], 'pos', 'neg', 'impos');

{% endhighlight %}

- (4) 生成正负样本数据文件，正样本文件格式：`**.jpg x1 y1 x2 y2`,其中(x1,x2)和(x2,y2)分别为正样本图片中检测目标的矩形位置的左上角和右下角的坐标位置，注意框定矩形位置时要稍微紧凑。负样本的格式为:`**.jpg`。
- (5) 正负样本的数据信息分别保存在两个`.txt`文件当中，以下是读取正样本图片数据的python代码：

{% highlight python %}
#coding:utf-8

import cv2
import os
import PIL

'''
Positive sample data format:
1.jpg 1 x1 y1 x2 y2
gump 2016-03-29

'''

path = 'F:/DPM/pos-new/' #image dir
doc = os.listdir(path)

refPt = []
IMGNA = ''
END = False


def button_click(event, x, y, flags, param):

    global refPt

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cv2.rectangle(im, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow(IMGNA, im)

for imgNa in doc:
    if not END:
        format = imgNa[imgNa.find('.'):]
        if ( format == '.jpg'):
            IMGNA = imgNa
            im = cv2.imread(imgNa)
            cv2.namedWindow(imgNa)
            cv2.setMouseCallback(imgNa, button_click)

            while True:

                cv2.imshow(imgNa, im)
                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    print imgNa, 1, refPt[0][0], refPt[0][1], refPt[1][0], refPt[1][1]
                    refPt = []
                    break
                if key == ord("q"):
                    END = True
                    break
    else:
        break

    cv2.destroyAllWindows()

cv2.destroyAllWindows()

{% endhighlight %}
---

- (6)修改`voc_config.m`文件代码，配置文件路径，其他相关路径根据代码运行提示来配置。

{% highlight c %}
% Parent directory that everything (model cache, VOCdevkit) is under
BASE_DIR    = '/home/gump/DOCU/voc-release5-New';

% PASCAL dataset year to use
PASCAL_YEAR = '2011';

% Models are stored in BASE_DIR/PROJECT/PASCAL_YEAR/
% e.g., /var/tmp/rbg/voc-release5/2007/
PROJECT     = 'UAV_result';
{% endhighlight %}

----

- (7)开始训练，打开matlab，输入：

> addpath(genpath('.'))   
> compile   
> pascal('uav', 3) %数字代表子模型个数   

---

- （8）生成结果(我大概训练了三个小时)，下图是六旋翼模型，生成完毕保存在`PROJECT`指定路径的文件下，比如以上则生成在`.\UAV_result\2011`路径下。现在就可以使用生成的`**_final.mat`模型进行目标检测了。

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/uav_model.png" style="width:400px">  

----


### 3.1 六旋翼监测效果，平均耗时1s左右。图1是对图片的识别，图2是加了kcf进行视频监测。


<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/uav_1.gif" style="width:500px">  

---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/uav_2.gif" style="width:500px">  
