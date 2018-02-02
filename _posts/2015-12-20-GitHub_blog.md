---
layout: post
title: 开篇-GitHub上搭建Blog批注
tags: [教程批注]
excerpt: "花了些许时间在GitHub上搭建了这个博客，以本文作为博客的开篇，附上如何在GitHub上搭建个人Blog批注。"
---

  花了些许时间在 GitHub 上搭建了这个博客，以本文作为博客的开篇，附上如何在 GitHub 上搭建个人 Blog 的批注。建立这个博客的初衷是为了记录自己在生活.学习.工作中的所见所闻，还有一个最重要的目的就是养成做批注并且做好记录的好习惯。愿能在此畅谈生活.谈工作. 谈技术，在生活和技术的旅途中愉悦前行。

---

Github Pages有以下几个优点：  
  1. 轻量级的博客系统，没有麻烦的配置
  2. 使用标记语言，比如Markdown 无需自己搭建服务器
  3. 根据Github的限制，对应的每个站有300MB空间
  4. 可以绑定自己的域名

当然它也有缺点：  
  1. 使用Jekyll模板系统，相当于静态页发布，适合博客，文档介绍等。
  2. 动态程序的部分相当局限，比如没有评论，不过还好有解决方案。
  3. 基于Git，很多东西需要动手，不像Wordpress有强大的后台

---

## 一. 基本步骤：

**(1)搭建UserName.github.io为域名的博客**   
- 第一步.创建一个名字为userName.github.io的公共仓库
- 第二步.把本地调试好的jekyll以master的分支上传到该仓库至此就能访问UserName.github.io的blog的。

**(2)绑定自己的blog.example.com的域名**    
- 第一步.去阿里云购买一个域名，并且以CNAME(别名)的方式绑定到userName.github.io
- 第二步.github的userName.github.io的仓库里添加一个CNMAE的文件里面只有一行：blog.example.com至此就能访问

**(3)搭建userName.github.io/repo为域名的博客**  
- 第一步.创建一个为repo(自己随意命名)的公共仓库
- 第二步.把本地调试好的jekyll以gp-pages的分支上传到该仓库至此就能访问userName.github.io/repo的blog了

**(4)让blog.example.com/repo指向该blog**
- 第一步.添加一个CNAME的文件指向blog.example.com就行了

**(5)配置ssh公钥push、代码高亮、markdown语言详解**

---

## 二.详细批注：

**1. 注册[GitHub](https://github.com/) 并且安装好[Git](https://desktop.github.com/)软件**

**2. 登录github添加一个远程仓库 repository**

<img src=" http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/1.png" style="width:450px">

- 添加一个Username.github.io的公共仓库;

<img src=" http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/2.png" style="width:450px">

**3. 本地搭建jekyll**

- 下载[Ruby 2.2.3 (x64)](http://rubyinstaller.org/downloads/)和[Ruby2.0 and (x64-64bits only)](http://rubyinstaller.org/downloads/);

<img src=" http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/3.png" style="width:450px">

<img src=" http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/4.png" style="width:450px">

- 安装Ruby,注意自己的安装路径,配置这个路径下面的./bin为PATH的环境变量，这样cmd就能调用Ruby了;
- 安装devikit，到该解压文件中执行cmd命令 ruby dk.rb init 

<img src=" http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/5.png" style="width:450px">

- 检查 ./config.yml中有ruby的安装路径没，若没有则填上;

<img src=" http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/6.png" style="width:450px">

- cmd执行命令 ruby dk.rb .install 

<img src=" http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/7.png" style="width:450px">

**4. 安装Gem**

- Git clone下来，命令 $ git clone http://github.com/rubygems/rubygems 

<img src=" http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/8.png" style="width:450px">

- 设置Gem到path环境变量，用Gem的安装./bin路径；

<img src=" http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/9.png" style="width:450px">

- 测试gem的环境变量和ruby的环境变量是否设置成功，cmd命令 ruby -version 和 gem -v ，分别有版本信息则安装成功;

<img src=" http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/10.png" style="width:450px">

- Gem官方网速很差，替换为淘宝的源，cmd命令:


> gem sources --add https://ruby.taobao.org/ --remove https://rubygems.org/

**5. gem安装jekyll**
- cmd安装命令

> gem install jekyll

<img src=" http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/11.png" style="width:450px">

- 测试是否安装成功

> jekyll -version   

<img src=" http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/12.png" style="width:450px">

**6. 本地调试**

- 到[jekyllThemes](http://jekyllthemes.org/)上下载一个自认为不错的模板;
- 解压该模板，在文件下执行命令cmd命令 jekyll serve ;
- 根据相关提示用 gem install 命令安装指定的依赖(注意查看模板工程文件下的README.MD文件);
- 打开浏览器访问该网址即可，类似 127.0.0.1:4000 (在命令窗口下有链接提示)；

<img src=" http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/13.png" style="width:450px">

**7. 然后push到github上去，到工程文件夹下面git bash,添加修改的文件时跳过init和add地址这两条语句:**


> $ git init                          
> $ git add .     
> $ git commit –m ‘first blood’     
> $ git remote add blog          (https://github.com/userNmae/userName.github.io.git(仓库地址))    
> $ git push blog master    

<img src=" http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/15.png" style="width:450px">

<img src=" http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/16.png" style="width:450px">

**8. config.yml文件配置**

> permalink: pretty   
> baseurl: ""   

**9. 域名绑定**

- 上传cname文件到工程根目录；

**10. ssh公钥来git(不需要每次git时输入密码)**

- 不能用 https 远程连接，要用 ssh ，将https改用ssh：

> $ git remote add UAV git@github.com:Gump-II/UAV-Detect.git
> $git push UAV master

- 使用命令 $ ssh-keygen -t rsa -C "your_email@youremail.com" 
- 三下enter，不输入密码;
- 本地windows账户下有.ssh文件，里面有 id_rsa两个文件，后缀为pub的则为公钥;
- 登陆gitgub账户，在仓库中的 setting>deploy 的keys中添加id_rsa.pub中的内容;
- 就可以欢快的用ssh来git了，不需要像https那样每次都要输入密码；
- 多个文件都使用公钥，建立一个公用的公钥，把单独仓库的公钥删除。

**11. 代码高亮**
<img src=" http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/BlogImages/14.png" style="width:450px">

**12. 添加评论**

- 使用[DISQUS](https://disqus.com/)评论服务(有时要被墙),具体使用参考官方帮助文档;

---
---
---
