---
layout: post
title: Openshift上搭建网站批注
tags:  [教程批注]
excerpt: "在Openshift上搭建个人网站，附上搭建批注                                    "
---

OpenShift是由全球开源解决方案领导者红帽公司（Redhat）在2011年5月推出的一个面向开源开发人员开放的平台即云服务(PaaS)。红帽OpenShift通过为开发人员提供在语言、框架和云上的更多的选择，使开发人员可以构建、测试、运行和管理他们的应用。OpenShift凭借创新的特性(包括CDI)领导PaaS市场，并支持Java EE 6，从而将PaaS的能力扩展到更丰富和更苛刻的应用。  
个人认为，红帽的openshift空间可以说是全球最优秀的免费空间，他有1G的大小没有任何广告，在我这里是秒开的，支持绑定域名，支持zend，cron等高级功能，完美的兼容wordpress。

---

### 1. 账号注册


- 在[Openshift](hhttps://www.openshift.com/)注册账号

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/openshift/1.png" style="width:450px">

### 2. 创建PHP应用

- 创建应用

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/openshift/2.png" style="width:450px">

- 选择PHP5.4

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/openshift/3.png" style="width:450px">

- 创建

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/openshift/4.png" style="width:450px">

### 3. 生成Putty密钥

- 下载密钥生成器[PuTTYgen](hhttp://www.chiark.greenend.org.uk/~sgtatham/putty/download.html)
- 密钥生成步骤并保存

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/openshift/5.png" style="width:450px">

- 将ssh复制到到所建的openshift应用窗口上

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/openshift/6.png" style="width:450px">

### 4. 找到自己中意的网站模板
- 模板下载[个人简历](hhttp://www.cssmoban.com/tags.asp?n=%E7%AE%80%E5%8E%86&n=web%E7%AE%80%E5%8E%86)

### 5. 登陆WinSCP

- 下载[WinSCP](hhttp://sourceforge.net/projects/winscp/)
- 打开网站应用

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/openshift/7.png" style="width:450px">

- 将ssh码copy到winscp上

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/openshift/8.png" style="width:450px">

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/openshift/9.png" style="width:450px">

- 点击winscp高级

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/openshift/10.png" style="width:450px">

- 密钥验证，将之前生成的密钥路径添加进去

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/openshift/11.png" style="width:450px">

- 登陆打开文件上传位置

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/openshift/12.png" style="width:450px">

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/openshift/13.png" style="width:450px">

- 将网站工程上传到此位置即可

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/openshift/14.png" style="width:450px">

### 6. 欢快的玩耍

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/openshift/15.png" style="width:450px">

### 7.教程文档下载

[教程文档下载](http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/data/openshift%E4%BD%BF%E7%94%A8%E6%95%99%E7%A8%8B.docx)

---
