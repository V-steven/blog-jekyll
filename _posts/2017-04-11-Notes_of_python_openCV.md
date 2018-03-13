---
layout: post
title: Notes of openCV-Python
tags:  [语言编程]
excerpt: "python有Numpy和matplotlib,其数据分析能力不逊于Matlab。被称为胶水语言的python，在丰富的接口下，特别是在Linux下，python和其他软件可以共同完成一个复杂的任务，例如：存储数据用Mysql，分析数据用R，展示数据用matplotlib，3D建模用OpenGL，GUI用Qt...... "
---
Python有Numpy和matplotlib,其数据分析能力不逊于Matlab。被称为胶水语言的python，在丰富的接口下，特别是在Linux下，python和其他软件可以共同完成一个复杂的任务，例如：存储数据用Mysql，分析数据用R，展示数据用matplotlib，3D建模用OpenGL，GUI用Qt，它们联合构成一个强大的工作流。虽然python非常强大，同时有自己的图像处理库PIL，但相对于OpenCV的图像处理能力还是比较弱小，其OpenCV含成熟的算法和丰富的函数。OpenCV提供了完善的python接口，那就用python结合OpenCV展现它们强大的图像处理能力，以下是[OpenCV-Python Tutoria](http://opencv-python-tutroals.readthedocs.org/en/latest/)的学习笔记。

---
---

**目录：1.GUI特性 2.核心操作 3.图像处理 4.特征提取 5.视屏分析 6.3D重构 7.机器学习 8.对象监测**

---
---

**1.1 picture**
{% highlight python %}
# -*- coding: utf-8 -*-

'Read/Show/Save image'

import numpy as np
import cv2

img = cv2.imread('img.jpg', cv2.IMREAD_COLOR) #read colorful image
cv2.namedWindow('Image', cv2.WINDOW_NORMAL) #adjustable windows
cv2.imshow('Image', img) #show image

key = cv2.waitKey(0) & 0xFF #64 bit system
if key == 27: #wait for ESC key to exit
    cv2.destroyAllWindows() #delete all windows
elif key == ord('s'): #wait for 's' key to save an exit
    cv2.imwrite('img2.png', img) #save
    cv2.destroyAllWindows()
{% endhighlight %}

----

**1.2 matplotlib**
{% highlight python %}
# _*_coding:utf-8 _*_

'How to use matplotlib'
#OpenCV是以(BGR)的顺序存储图像数据的
#Matplotlib是以(RGB)的顺序显示图像的
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg', 0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([]) #to hide tick values on X and Y axis
plt.show()& 0xFF
{% endhighlight %}

---

**1.3 video**
{% highlight python %}
# -*- coding: utf-8 -*-

'Capture a frame from video'

import numpy as np
import cv2

cap = cv2.VideoCapture(0) #open camera

while True:
	ret, frame = cap.read() # capture frame by frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #read gray image from a frame
	cv2.imshow('frame', gray)
	if cv2.waitKey(0) & 0xFF == ord('q'):
		break

cap.release() #release the capture
cv2.destroyAllWindows()
-------------------------------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

'Play video from file'

import numpy as np
import cv2

cap = cv2.VideoCapture('test.avi') #can also capture video from camera

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.IMREAD_COLOR) #cv2.COLOR_BGR2GRAY

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
-------------------------------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

'Save Video'

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

fourcc = cv2.cv.CV_FOURCC(* 'DIVX') #windows
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480)) #save pattern

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame, 1) #'0' image reversal
    
        out.write(frame) #save video
    
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #delay
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
{% endhighlight %}

---

**1.4 draw**
{% highlight python %}
# -*- coding: utf-8 -*-

'Draw a line,rectangle,circle,ellipse'

import numpy as np
import cv2

img = np.zeros((512, 512, 3), np.uint8) #empty color image
cv2.line(img,(0,0),(420,420),(255,0,0),3) #line
cv2.circle(img, (450, 70), 70, (0,0,255), -1) #circle
cv2.rectangle(img,(385,0),(510,120),(0,255,0),3) #rectangle
cv2.ellipse(img,(255,255),(100,50),0,0,180,255,-1) #ellipse


cv2.imshow('Draw', img)
cv2.waitKey(0)
cv2.destroyWindow('Draw')
{% endhighlight %}
---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/draw.png" style="width:500px">

---

**1.5 mouse**
{% highlight python %}
# -*- coding: utf-8 -*-

'Mouse callback events'

import cv2
import numpy as np

press = False # press down left button of the mouse :true
mode = True # True: rectangle ,False: circle
ix, iy = -1, -1

def draw(event, x, y, flags, param): #mouse callback function
	global ix, iy, press, mode

	if event == cv2.EVENT_LBUTTONDOWN: # button down
		press = True
		ix, iy = x, y # 'button down' event, start point
	
	elif event == cv2.EVENT_MOUSEMOVE: # mouse move
		if press == True:
			if mode == True:
				cv2.rectangle(img, (ix, iy), (x, y), (0,255,0), -1) # rectangle
			else:
				cv2.circle(img, (x, y), 5, (0, 0, 255), -1) # circle
	
	elif event ==cv2.EVENT_LBUTTONUP: # button up
		press = False
		if mode == True:
			cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
		else:
			cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

img = np.zeros((500,500,4), np.uint8) #empty color image
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        mode = not mode
    elif k == 27:
        break

cv2.destroyAllWindows()

{% endhighlight %}

---

**1.6 trackbar**
{% highlight python %}
# -*- coding: utf-8 -*-

' Trackbar as the color palette'

import cv2
import numpy as np

def nothing (x): # do nothing
    pass

img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('image')

cv2.createTrackbar('R', 'image', 0, 255, nothing) # create trackbar, int value
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)

switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image', 0, 1, nothing)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    
    r = cv2.getTrackbarPos('R', 'image') # get value
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    s = cv2.getTrackbarPos(switch, 'image')
    
    if s == 0: # switch
        img[:] = 0
    else:
        img[:] = [b,g,r]

cv2.destroyAllWindows()
{% endhighlight %}

---

**1.7 RGB & BGR**
{% highlight python %}
# _*_ coding:utf-8_*_

'BGR & RGB'

import cv2
import numpy as np
import matplotlib.pyplot as plt

#OpenCV follows BGR order, while matplotlib follows RGB order
img = cv2.imread('img1.jpg')
b,g,r = cv2.split(img)
img2 = cv2.merge([r,g,b])
plt.subplot(121);plt.title('BGR'),plt.imshow(img) # expects distorted color (BGR)
plt.subplot(122);plt.title('RGB'),plt.imshow(img2) # expect true color (RGB)
plt.show()

cv2.imshow('BGR',img) # expects true color (BGR)
cv2.imshow('RGB',img2) # expects distorted color (RGB)
cv2.waitKey(0)
cv2.destroyAllWindows()
{% endhighlight %}
---
<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/matplotlib.png" style="width:500px">

---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/cv2.png" style="width:500px">

---

**2.1 operation**
{% highlight python %}
# -*- coding: utf-8 -*-

'Basic operation of image'

import numpy as np
import cv2

img  = cv2.imread('img.jpg')
img[100,100] = [255,255,255] # change RGB
img.itemset((10,10,2),100) # anothor the way of change RGB
blue = img[100,100,1] # get value of one channel

# img[row,column,channel],
# 灰度就是没有色彩，RGB色彩分量全部相等

ball=img[40:240,230:290] # modify specific area
img[0:200,100:160]=ball

b, g, r = cv2.split(img) # split channel
img = cv2.merge((b, g, r)) # merge channel

img[:,:,1] = 0 #change value of one of the channels

print img.size # size
print img.shape # shape
print img.dtype # type

cv2.imshow('image', img)

if cv2.waitKey(0) & 0xFF == 27:
	cv2.destroyAllWindows()

{% endhighlight %}

---

**2.2 arithmetic**

{% highlight python %}
# -*- coding: utf-8 -*-

'Image Arithmetic Operation'

import cv2
import numpy as np

img1 = cv2.imread('img.jpg')
img2 = cv2.imread('img1.jpg')

dst = cv2.addWeighted(img1, 0.9, img2, 0.1, 0)  # weight

cv2.imshow('image', dst)
cv2.imwrite('img_dst.jpg',dst)
cv2.waitKey(0)
cv2.destoryAllWindow()

{% endhighlight %}

---

**2.3 bitwise**
{% highlight python %}
# -*- coding: utf-8 -*-

'Bitwise Operations: Add logo'

import cv2
import numpy as np

img1 = cv2.imread('img.jpg')
img2 = cv2.imread('logo.png')

#put logo on top-left corner,create ROI
rows, cols, channels = img2.shape
roi = img1[0 : rows, 0 : cols]

#create a mask of logo and create its inverse mask also
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # gray

# mask: target area->1 ,other area->0
ret, mask = cv2.threshold(img2_gray, 20, 255, cv2.THRESH_BINARY) #binary, 20:threshold value
mask_inv = cv2.bitwise_not(mask)# bit not

img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv) # bit and, roi and mask
img2_fg = cv2.bitwise_and(img2, img2, mask = mask)

dst = cv2.add(img1_bg, img2_fg) # add
img1[0 : rows, 0 : cols] = dst # add logo

cv2.imshow('image', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
{% endhighlight %}

---

**3.1 colorSpaces**

{% highlight python%}
# -*- coding: utf-8 -*-

'Changing ColorSpaces'

import cv2

img = cv2.imread('img.jpg')
GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #BGR-> GRAY
HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #BGR-> HSV
# H:色彩[0,179]; S:饱和度[0,255]; V:亮度[0,255]

cv2.imshow('HSV', HSV)
cv2.imshow('GRAY', GRAY)

cv2.waitKey(0)
cv2.destroyAllWindows()
{% endhighlight %}

---

**3.2 object track**
{% highlight python %}
# _*_coding:utf-8_*_

'Object Track'

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:

    ret,frame = cap.read()

    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 43, 46]) # threshold value of blue
    upper_blue = np.array([124, 255, 255])
    
    mask = cv2.inRange(HSV, lower_blue, upper_blue) # create a mask by inRange()
    
    res = cv2.bitwise_and(frame, frame, mask = mask) # bit and
    
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    
    if cv2.waitKey(5) & 0xFF ==27:
        break
cv2.destroyAllWindows()
{% endhighlight %}

---

**3.3.1 rotation/translation/scaling**
{% highlight python %}
# _*_coding:utf-8_*_

'Translation/Rotation/Scaling'

import cv2
import numpy as np

img = cv2.imread('img.jpg', 0)
rows, cols = img.shape[:2]

# M1:平移矩阵，满足[[1, 0 ,tx], [0, 1, ty]]
# M2:旋转矩阵，openCV中旋转矩阵修改了而非[[cos, -sin], [sin, cos]]
M1 = np.float32([[1, 0, 100], [0, 1, 50]]) #translation
M2 = cv2.getRotationMatrix2D((cols / 2,rows / 2), 45, 0.6) #rotate,(center point, angle,  zoom)

# warpAffine(src, M, dsize) src:源图像， M:矩阵, dsize:平移后的大小
dst_tra = cv2.warpAffine(img, M1, (cols, rows)) #平移
dst_rot = cv2.warpAffine(img, M2, (2 * cols, 2 * rows)) #旋转
dst_sca = cv2.resize(img, (2 * cols, 2 * rows), interpolation = cv2.INTER_CUBIC) #缩放

cv2.imshow('img_tra', dst_tra)
cv2.imshow('img_rot', dst_rot)
cv2.imshow('img_sca', dst_sca)
cv2.waitKey(0)
cv2.destroyAllWindows()
{% endhighlight %}

---

**3.3.2 affine transformation**
{% highlight python %}
# _*_coding:utf-8_*_

'Affine Transform'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

M = cv2.getAffineTransform(pts1,pts2)
# 仿射：原图中所有的平行线在结果图像中同样平行。

dst = cv2.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Input'),plt.axis('off')
plt.subplot(122),plt.imshow(cv2.cvtColor(dist, cv2.COLOR_BGR2RGB))
plt.title('Output'),plt.axis('off')
plt.show()
{% endhighlight %}

---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/affine.png" style="width:500px">

---

**3.3.3 perspective transformation**

{% highlight python %}
# _*_ coding:utf-8_*_

'Perspective Transformation'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg')
rows, cols, ch = img.shape[:3]

pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

M = cv2.getPerspectiveTransform(pts1, pts2)
# 透视面绕迹线旋转，几何图形不变
dst = cv2.warpPerspective(img, M, (300, 300))#透视变换

plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()
{% endhighlight %}
---
<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/perspective.png" style="width:500px">

---

**3.4.1 simple thresholding**
{% highlight python %}
# _*_coding:utf-8_*_

'Simple Threshold'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg', 0)

#cv2.threshold(img, thresh, maxval,type)
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) # src>thresh ->maxval, otherwise ->0
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV) # src>thresh ->0, otherwise ->maxval
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC) # src>thresh ->threshold, otherwise ->src
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO) # src>thresh ->src, otherwise ->0
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV) #src>thresh ->0, otherwise ->src

titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in xrange(6):
	plt.subplot(2, 3, i + 1)
	plt.imshow(images[i], 'gray')
	plt.title(titles[i])
	plt.xticks([])
	plt.yticks([])
plt.show()
{% endhighlight %}
---
<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/threshold.png" style="width:500px">

---

**3.4.2 adaptive thresholding**

{% highlight python %}
# _*_coding:utf-8_*_

'Adaptive Threshold'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg', 0)
img = cv2.medianBlur(img, 5)

# cv2.ADAPTIVE_THRESH_MEAN_C：阈值附近区域的平均值。
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C：阈值附近的值，其中权重是高斯窗口的加权和。

ret,th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in xrange(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
{% endhighlight 5%}
---
<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/ada_threshold.jpg" style="width:500px">

---

**3.4.3 Ostu's binariztion**
{% highlight python %}
# _*_coding:utf-8_*_

'Otsu's Binarization'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg',0)

# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

for i in xrange(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
{% endhighlight %}
---
<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/otsu.jpg" style="width:500px">

---

**3.5.1 filter2D**
{% highlight python %}
# _*_ coding:utf-8_*_

'Filter2D'
#低通滤波LPF：去噪，模糊
#高通滤波HPF：确定图像边缘
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg')

kernel = np.ones((5, 5), np.float) / 25
dst = cv2.filter2D(img, -1, kernel) # 二维卷积

cv2.imshow('Image', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
{% endhighlight %}

---

**3.5.2 average/gaussian/median/biateral**
{% highlight python %}
# _*_ coding:utf-8_*_

'Average/Gaussian/Median/Biateral'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img1.jpg')

img_average = cv2.blur(img, (5, 5)) # 平均滤波
img_gaussian = cv2.GaussianBlur(img, (5, 5), 0) # 高斯滤波
img_median = cv2.medianBlur(img, 5) # 终止滤波
img_biateral = cv2.bilateralFilter(img, 9, 75, 75) #双边滤波

titles = ['Original Image', 'Average', 'Gaussian Filter', 'Median Fiter', 'Biateral Fiter']
images = [img, img_average, img_gaussian, img_median, img_biateral]

for i in xrange(5):
	plt.subplot(2, 3, i + 1)
	#plt.imshow(images[i]) # RGB顺序显示图像
	plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)) #BGR 显示图像, cv2.imshow()一样
	plt.title(titles[i])
	plt.axis("off") # plt.xticks([]), plt.yticks([])
plt.show()
{% endhighlight %}
---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/average.png" style="width:500px">

---

**3.6.1 erosion/dilation**
{% highlight python %}
# _*_ coding:utf-8_*_

'Morphological Transformations: erosion/dilation'

import cv2
import numpy as np
# operate binary images
img = cv2.imread('img3.png',0 )
kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations = 1)
dilation = cv2.dilate(img, kernel, iterations = 1)

cv2.imshow('Erosion', erosion) # 腐蚀
cv2.imshow('Dilation', dilation) # 膨胀
cv2.waitKey(0)
cv2.destroyAllWindows()
{% endhighlight %}
---
<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/erosion.png" style="width:500px">

---
**3.6.2 opening/closing/gradient/tophat/blackhat**
{% highlight python %}
# _*_ coding:utf-8_*_

'Opening/Closing/Gradient/Tophat/Blackhat'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img3.png', 0)

kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) # 开运算
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) # 闭运算
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel) # 形态学梯度
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel) # 礼帽
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel) # 黑帽

titles = ['Orginal', 'Opening', 'Closing', 'Gradient', 'Tophat', 'Blackhat']
images = [img, opening, closing, gradient, tophat, blackhat]

cv2.imshow('i', opening)

for i in xrange(6):
	cv2.imshow(titles[i], images[i])
cv2.waitKey(0)
cv2.destroyAllWindows()
{% endhighlight %}
---
<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/opening.png" style="width:500px">

---
**3.7.1 sobel**
{% highlight python %}
# _*_coding:utf-8_*_

'Image Gradients:Sobel'

#图像梯度，图像边界

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('111.png', 0)

# Output dtype = cv2.CV_8U
sobelx8u = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize = 5 )
# 也可以将参数设为-1

# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
sobelx64f = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

titles = ['img', 'sobelx8u', 'sobel_8u']
images = [img, sobelx8u, sobel_8u]

for i in xrange(3):
	plt.subplot(1, 3, i + 1)
	plt.imshow(images[i], 'gray')
	plt.title(titles[i])
	plt.axis("off")
plt.show()
{% endhighlight %}

---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/sobel.png" style="width:500px">

---

**3.7.2 laplacian**
{% highlight python %}
# _*_coding:utf-8_*_

'Image Gradients:Laplacian'

#图像梯度，图像边界

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('11.png', 0)

laplacian = cv2.Laplacian(img, -1) # 图像的数据类型
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 5) # x方向一阶导 ksize = -1为Scharr
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize = 5) # y方向一阶导

gradient = cv2.subtract(sobelx, sobely)
#gradient = cv2.convertScaleAbs(gradient)


titles = ['Original', 'Laplacian', 'Sobelx', 'Sobely', 'gradient']
images = [img,  laplacian, sobelx, sobely, gradient]

for i in xrange(5):
	plt.subplot(2, 3, i + 1)
	plt.imshow(images[i], 'gray')
	plt.title(titles[i])
	plt.axis("off")
plt.show()
{% endhighlight %}
---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/laplacian.png" style="width:500px">

---

**3.8 canny**
{% highlight python %}
# _*_coding:utf-8_*_

'Canny edge detection'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg', 0)
edge = cv2.Canny(img, 50, 140) # min max

titles = ['Origial', 'Edge']
images = [img, edge]

for i in xrange(2):
	plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
	plt.title(titles[i]), plt.axis('off')

plt.show()
{% endhighlight %}
---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/edge.png" style="width:500px">

---

**3.9.1 contours start**
{% highlight python %}
# _*_coding:utf-8_*_

'Contours'

import cv2  

img = cv2.imread('contours.jpg')  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY) # 二值化

image, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (255,0,0), 2)

cv2.imshow("img", img)  
cv2.waitKey(0)
{% endhighlight %}
---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/contours.png" style="width:500px">

---

**3.9.2 contours feature1**
{% highlight python %}
# _*_coding:utf-8_*_

'M/Area/Perimeter/Approx/Hull'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('contours2.png')
img1 = cv2.imread('contours2.png')
img2 = cv2.imread('contours2.png')
img3 = cv2.imread('contours2.png')

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray, 127, 255, 0)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[-1]
M = cv2.moments(cnt) # 重心
area = cv2.contourArea(cnt) # 面积
perimeter = cv2.arcLength(cnt, True) # 周长

epsilon = cv2.arcLength(cnt, True)
approx1 = cv2.approxPolyDP(cnt, 0.1 * epsilon, True)
approx2 = cv2.approxPolyDP(cnt, 0.01 * epsilon, True)

hull = cv2.convexHull(cnt)

cv2.drawContours(img,approx1,-1,(255, 0, 0),14)
cv2.drawContours(img1,approx2,-1,(255, 0, 0),14)
cv2.drawContours(img2,hull,-1,(255, 0, 0),14)

titles = ['Origial', '0.1 Approx','0.01 Approx', 'Hull']
images = [img3, img, img1, img2]

for i in xrange(4):
	plt.subplot(1 ,4, i + 1)
	plt.title(titles[i])
	plt.imshow(images[i])
	plt.axis('off')
plt.show()


{% endhighlight %}
----

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/approx.png" style="width:500px">

---

**3.9.3 contours feature2**
{% highlight python %}
# _*_coding:utf-8_*_

'Rectangle/MinAreaRect/Circle/Ellipse/Line'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('contours1.png')
img1 = cv2.imread('contours1.png')
img2 = cv2.imread('contours1.png')
img3 = cv2.imread('contours1.png')
img4 = cv2.imread('contours1.png')
img5 = cv2.imread('contours1.png')

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray, 127, 255, 0)
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]

x,y,w,h = cv2.boundingRect(cnt) # rectangle
img_rect = cv2.rectangle(img1, (x, y), (x + w, y + h), (0,255,0), 2)

rect = cv2.minAreaRect(cnt) # minAreaRect
box = cv2.boxPoints(rect)
box = np.int0(box)
img_box = cv2.drawContours(img2,[box],0,(0,255,255),2)

(x,y),radius = cv2.minEnclosingCircle(cnt) #circle
center = (int(x),int(y))
radius = int(radius)
img_cir = cv2.circle(img3,center,radius,(255,255,0),2)

ellipse = cv2.fitEllipse(cnt) #ellipse
img_ell = cv2.ellipse(img4,ellipse,(0,0,255),2)

rows,cols = img.shape[:2] # line
[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx)+y)
img_line = cv2.line(img, (cols - 1,righty), (0, lefty), (255, 0, 0), 2)

titles = ['Origial', 'minAreaRect', 'Box', 'Circle', 'Ellipse', 'Line']
images = [img, img_rect, img_box, img_cir, img_ell, img_line]


for i in xrange(6):
	plt.subplot(2, 3, i + 1)
	plt.title(titles[i])
	plt.imshow(images[i])
	plt.axis('off')
plt.show()
{% endhighlight %}
---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/contours1.png" style="width:500px">

---

**3.9.4 convexity defects**
{% highlight python %}
# _*_coding:utf-8_*_

'Convexity Defects'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('contours1.png')
img1 = cv2.imread('contours1.png')

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 127, 255,0)
image, contours, hierarchy = cv2.findContours(thresh,2,1)

cnt = contours[0]
hull = cv2.convexHull(cnt,returnPoints = False)
defects = cv2.convexityDefects(cnt,hull)

for i in range(defects.shape[0]):
	s,e,f,d = defects[i,0]
	start = tuple(cnt[s][0])
	end = tuple(cnt[e][0])
	far = tuple(cnt[f][0])
	cv2.line(img,start,end,[0,255,0],2)
	cv2.circle(img1,far,5,[0,0,255],-1)

titles = ['Line', 'Circle']
images = [img, img1]

for i in xrange(2):
	plt.subplot(1, 2, i + 1)
	plt.imshow(images[i])
	plt.title(titles[i])
	plt.axis('off')
plt.show()
{% endhighlight %}
---   

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/defects.png" style="width:500px">   

---

**3.9.5 match shapes**
{% highlight python %}
# _*_coding:utf-8_*_

'Match Shapes'

import cv2
import numpy as np

img1 = cv2.imread('1.png', 0)
img2 = cv2.imread('2.png', 0)
img3 = cv2.imread('3.png', 0)

ret, thresh1 = cv2.threshold(img1, 127, 255, 0)
ret, thresh2 = cv2.threshold(img2, 127, 255, 0)
ret, thresh3 = cv2.threshold(img3, 127, 255, 0)

images, contours, hierarchy = cv2.findContours(thresh1, 2, 1)
cnt1 = contours[0]

images, contours, hierarchy = cv2.findContours(thresh2, 2, 1)
cnt2 = contours[0]

images, contours, hirearchy = cv2.findContours(thresh3, 2, 1)
cnt3 = contours[0]

ret1 = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
ret2 = cv2.matchShapes(cnt1, cnt3, 1, 0.0)

print ret1, ret2
{% endhighlight %}

---

**3.10.1 plotting histograms**
{% highlight python %}
# _*_coding:utf-8_*_

'Plotting Histograms'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img_gray = cv2.imread('img.jpg', 0)
img_color = cv2.imread('img.jpg', 1)

color = ('b', 'g', 'r')

plt.subplot(1,2,1)
for i,col in enumerate(color):  #同时遍历数组和索引
    histr = cv2.calcHist([img_color],[i],None,[256],[0,256]) # BGR plot
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.title('Histograms Gray')

plt.subplot(1,2,2)
plt.hist(img_gray.ravel(), 256, [0, 256]); # gray plot
plt.title('Histograms Color')
plt.show()
{% endhighlight %}
---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/plot.png" style="width:500px">

---

**3.10.2 histograms region**
{% highlight python %}
# _*_coding:utf-8_*_

'Histograms of some region of an image'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg', 0)

#create a mask
mask = np.zeros(img.shape[0:2], np.uint8)
mask[50 : 300, 50 : 300] = 255
img_mask = cv2.bitwise_and(img, img, mask = mask)

hist_full = cv2.calcHist([img], [0], None, [256], [0, 256]) # without mask
hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256]) # with mask\

titles = ['Origial', 'Mask', 'Mask image', 'histograms']
images = [img, mask, img_mask, hist_full, hist_mask]

for i in xrange(4):
	plt.subplot(2, 2, i + 1)
	plt.title(titles[i])
	if i == 3:
		plt.plot(images[i])
		plt.plot(images[i + 1])
	else:
		plt.imshow(images[i], 'gray')
		plt.axis('off')
plt.show()
{% endhighlight %}
---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/hisMask.png" style="width:500px">  

---

**3.10.3 histogram equlazation**
{% highlight python %}
# _*_coding:utf-8_*_

'Histogram Equalization'
#高质量的像素值分布更加广泛
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg', 0)

hist, bins = np.histogram(img.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_mormalized = cdf * hist.max() / cdf.max()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')
img1 = cdf[img]

titles = ['Origial Image', 'Equalization Image', 'Origial Historgram', 'Equalization Historgram']
images = [img, img1, img, img1]

for i in xrange(4):
	plt.subplot(2, 2, i + 1)
	plt.title(titles[i])
	if i > 1:
	
		plt.plot(cdf_mormalized, color = 'b')
		plt.hist(images[i].flatten(), 256, [0, 256], color = 'r')
		plt.xlim([0, 256])
		plt.legend(('cdf', 'historgram'), loc = 'upper left')
	
	else:
		plt.imshow(images[i], 'gray')
		plt.axis('off')
plt.show()
{% endhighlight %}
---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/equalization.png" style="width:500px">  

---

**3.10.4 Adaptive Histogram Equalization**
{% highlight python %}
# _*_coding:utf-8_*_

'CLAHE (Contrast Limited Adaptive Histogram Equalization)'

#避免改变图片的对比度而丢失细节信息
#图片的多个小块分别均质化再缝合(双线性差值)

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('cli.png', 0)

clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
cl1 = clahe.apply(img)

titles = ['Original', 'CLAHE']
images = [img, cl1]

for i in xrange(2):
	plt.subplot(1 ,2, i + 1)
	plt.imshow(images[i], 'gray')
	plt.title(titles[i])
	plt.axis('off')
plt.show()
{% endhighlight %}
---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/clahe.png" style="width:500px">  

---

**3.10.5 Plotting 2D Historgrams**
{% highlight python %}
# _*_coding:utf-8_*_

'Plotting 2D Histograms'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Histogram')
plt.imshow(hist, interpolation = 'nearest')

plt.show()
{% endhighlight %}
---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/2DHist.png" style="width:500px">  

---

**3.11.1 Fourier Transform in Numpy**
{% highlight python %}
# _*_coding:utf-8_*_

'Fourier Transform in Numpy'

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg', 0)

#构建振幅谱(magnitude spectrum)
#振幅谱：中心更加白亮则说明低频分量多
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitudeSpectrum = 20 * np.log(np.abs(fshift))

#频域变换
rows, cols = img.shape
crow,ccol = rows/2 , cols/2
fshift[crow - 30 : crow + 30, ccol - 30 : ccol + 30] = 0
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

titles = ['Original Image', 'Magnitude Spectrum', 'JET']
images = [img, magnitudeSpectrum, img_back]

for i in xrange(3):
	plt.subplot(1, 3, i + 1)
	plt.imshow(images[i], 'gray')
	plt.title(titles[i])
	plt.axis('off')
plt.show()
{% endhighlight %}
---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/fourierNumpy.png" style="width:500px">  

---

**3.11.2 Fourier Transform in OpenCV**
{% highlight python %}
# _*_coding:utf-8_*_

'Fourier Transform in OpenCV'

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('img.jpg', 0)

dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitudeSpectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

rows, cols = img.shape
crow, ccol = rows / 2, cols / 2

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows,cols,2),np.uint8)
mask[crow-30:crow+30, ccol-30:ccol + 30] = 1

# apply mask and inverse DFT
fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

titles = ['Original Image', 'Magnitude Spectrum', 'JET']
images = [img, magnitudeSpectrum, img_back]

for i in xrange(3):
	plt.subplot(1, 3, i + 1)
	plt.title(titles[i])
	plt.imshow(images[i], 'gray')
	plt.axis('off')
plt.show()
{% endhighlight %}
---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/fourierOpencv.png" style="width:500px">  

---

**3.11.3 HPF or LPF**
{% highlight python %}
# _*_coding:utf-8_*_

'HPF or LPF'

import cv2
import numpy as np
from matplotlib import pyplot as plt

mean_filter = np.ones((3, 3))

# creating a guassian filter
x = cv2.getGaussianKernel(5, 10)
#x.T 为矩阵转置
gaussian = x * x.T

# different edge detecting filters

# scharr in x-direction
scharr = np.array([[-3, 0, 3],[-10, 0, 10],[-3, 0, 3]])

# sobel in x direction
sobel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])

# sobel in y direction
sobel_y= np.array([[-1,-2,-1],[0, 0, 0],[1, 2, 1]])

# laplacian
laplacian=np.array([[0, 1, 0],[1,-4, 1],[0, 1, 0]])

filters = [mean_filter, gaussian, laplacian, sobel_x, sobel_y, scharr]
filter_name = ['mean_filter', 'gaussian','laplacian', 'sobel_x', 'sobel_y', 'scharr_x']
fft_filters = [np.fft.fft2(x) for x in filters]
fft_shift = [np.fft.fftshift(y) for y in fft_filters]
mag_spectrum = [np.log(np.abs(z) + 1) for z in fft_shift]

for i in xrange(6):
	plt.subplot(2,3,i+1),plt.imshow(mag_spectrum[i], cmap = 'gray')
	plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
plt.show()
# 从以下图像可以判断是高通还是低通

{% endhighlight %}
---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/HPF.png" style="width:500px">  

---
