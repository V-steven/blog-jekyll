---
layout: post
title: Kernelized Correlation Filters
tags:  [图像识别]
excerpt: "Kernelized Correlation Filters跟踪算法                                 "
---

KCF是一种鉴别式追踪方法，这类方法一般都是在追踪过程中训练一个目标检测器，使用目标检测器去检测下一帧预测位置是否是目标，然后再使用新检测结果去更新训练集进而更新目标检测器。而在训练目标检测器时一般选取目标区域为正样本，目标的周围区域为负样本，当然越靠近目标的区域为正样本的可能性越大。

**(1)环境：VS2013/windows10**

**(2)C++[[code]](http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/data/KCF-Tracker.rar)**

**(3)Kernelized Correlation Filters[网站](http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/data/KCF-Tracker.rar)，有源码下载**

---

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Img/KCF.gif" style="width:800px">  


---
