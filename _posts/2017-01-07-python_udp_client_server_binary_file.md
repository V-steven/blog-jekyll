---
layout: post
title: python send and receive binary files(UDP)
tags:  [语言编程]
excerpt: "Python用socket通讯(UDP),以二进制图片的收发为例，实现对图片的解包和打包，通过UDP的传输协议进行传输。   "
---
Python用socket通讯(UDP),以二进制图片的收发为例，实现对图片的解包和打包，通过UDP的传输协议进行传输。

---

**1.图片的打包和解包**

- 使用了struct对二进制文件的打包和解包,官方[数据类型](https://docs.python.org/3/library/struct.html#format-characters)。
- 涉及文件操作。

{% highlight python %}
# -*- coding: utf-8 -*-
import os
import struct
import stat

fileName = 'img1.jpeg'
fileFir = open(fileName, 'rb')
fileSize = os.stat(fileName)[stat.ST_SIZE] #read the size of the image,in bytes
chList = []
string = ""

for i in range(0, fileSize):
    ch, = struct.unpack("c", fileFir.read(1)) #return tuple,read 1 byte,
    chList.append(ch)

for i in range(0,fileSize):
    string = string + struct.pack("c", chList[i]) #pack string

fileSec=open("img2.jpeg",'wb')
fileSec.write(string)
fileFir.close()
fileSec.close()
{% endhighlight %}

---

**2.socket通讯(UDP)，二进制收发**

- UDP无连接传输,没有TCP复杂,三次握手,没有错误重传机制,发的只管发,收得只管收,常使用在对数据帧要求不高的地方。
- server(1)端建立数据报形式的socket(2)公开一个端口,客户端连接(3)接收数据。
- client(1)新建数据报socket(2)发送数据。

{% highlight python %}
# -*- coding: utf-8 -*

'python udp server get binary file.'

import socket

port=8888
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM) #DGRAM -> UDP
s.bind(('localhost',port))

f=open("img2.jpeg",'wb')
datas = ""
state = 0

while True:
    data,addr=s.recvfrom(1024)# LAN 1024 bytes
    if state==0:
        if data=="bin": #flag of beginning
            state=1 #start receiving
            datas=""
            print("start receiving....")
    elif state==1:
        if data=="end": #flag of end
            print("transmit done....")
            break
        else:
            datas += data
f.write(datas)
f.close()
print("saved to img2.jpeg")
--------------------------------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

'python udp client send binary file.'

import socket
import os
import stat  # get file size
import struct  # pack binary
MAX_PACK_SIZE = 1024
DEST_IP = 'localhost'
DEST_PORT = 8888

filename = "img1.jpeg"
def send_file(client, filename):
    filesize = os.stat(filename)[stat.ST_SIZE]
    print("%s size: %d Bytes" % (filename, filesize))
    f = open(filename, "rb")
    chList = []
    for i in range(0, filesize):
        (ch,) = struct.unpack("B", f.read(1))
        chList.append(ch)
    client.sendto("bin", (DEST_IP, DEST_PORT)) #send "bin" string
    packSize = 0
    string = ""
    for i in range(0, filesize):
        packSize = packSize + 1
        string = string + struct.pack("B", chList[i])
        if (MAX_PACK_SIZE == packSize or i == filesize - 1):
            client.sendto(string, (DEST_IP, DEST_PORT)) # send datas
            packSize = 0
            string = ""
    client.sendto("end", (DEST_IP, DEST_PORT)) #send "end" string

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_file(client,filename)
client.close()
{% endhighlight %}

---
