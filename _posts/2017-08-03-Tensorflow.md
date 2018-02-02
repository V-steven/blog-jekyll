---
layout: post
title: How to configure tensorflow in manjaro
tags:  [深度学习]
excerpt: "在manjaro上配置tensorflow的详细教程"
---


**1.安装Cuda**

**2.安装Tensorflow**

**3.安装Jupyter notebook**

---
---

## 1.安装Cuda 

- (1)显卡是GTX1070，安装cuda8.0
{% highlight c %}
yaourt cuda
{% endhighlight %} 
- (2)安装的包在'/opt/cuda'
- (3)添加环境变量
  {% highlight c %}
  export GLPATH="/usr/lib64/nvidia"
  export GLLINK="-L/usr/lib64/nvidia"
  export DFLT_PATH="/usr/lib64"
  {% endhighlight %}
- (4)运行例子，打开cuda目录下的sample，例：在bindlessTexture下make，生成bindlessTexture文件
  {% highlight Python %}
  ./bindlessTexture
  {% endhighlight %}
---

## 2.安装Tensorflow
- (1)安装Python3和pip3
- (2)安装python-virtualenv
- (3)配置virtualenv环境
  {% highlight c %}
  virtualenv --system-site-packages -p python3 tensorflow
  {% endhighlight %}
- (4)激活virtualenv环境
  {% highlight c %}
  source ./tensorflow/bin/activate
  {% endhighlight %}
- (5)在环境下安装tensorflow
  {% highlight c %}
  pip3 install --upgrade tensorflow-gpu 
  {% endhighlight %}
---

## 3.安装Jupyter notebook
- (1)安装Jupyter notebook
  {% highlight c %}
  sudo python3 -m pip install jupyterhub notebook ipykernel
  sudo python3 -m ipykernel install
  {% endhighlight %}
- (2)配置环境，精简激活命令
  {% highlight c %}
  alias tf="source /tensorflow/bin/activate"
  {% endhighlight %}
---
<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/1.png" style="width:600px">
