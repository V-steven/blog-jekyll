---
layout: post
title: python实现socket通讯(TCP和UDP)
tags:  [语言编程]
excerpt: "python实现socket通讯，分别从TCP和UDP协议进行分析                                                   "
---
如前面两篇[python send and receive binary files(UDP)](http://blog.ganyutao.com/python/2016/01/07/python_udp_client_server_binary_file/)和 [Python入门Web server](http://blog.ganyutao.com/python/2016/01/03/Pyhonr_web_server/)都涉及到了socket应用，这篇对tcp和udp编程简单解说。

---

**1.TCP**

- sever(1)创建一个socket(2)公开绑定的端口地址(3)端口监听(4)accept等待连接(5)IO操作。
- client(1)创建一个socket(2)获得主机连接(3)IO操作。

{% highlight python %}
# -*- coding: utf-8 -*-

'tcpServer'

import socket

IP, PORT = 'localhost', 8888
MAX_PACK_SIZE = 1024

def tcpServer():   
    srvSock = socket.socket( socket.AF_INET, socket.SOCK_STREAM) #socket   
    srvSock.bind((IP, PORT))  #bind ip and port
    srvSock.listen(5)  #listen
    
    while True:  
        cliSock, (remoteHost, remotePort) = srvSock.accept()   
        #print "[%s:%s] connected" % (remoteHost, remotePort)
        cliSock.send("hello client!") #send    
    
        dat = cliSock.recv(MAX_PACK_SIZE)  #backup and receive message
       	if dat:
       		print dat
       	else:
       		break
    
        cliSock.close()   

if __name__ == "__main__":   
    tcpServer()  
{% endhighlight %}

---
{% highlight python %}
# -*- coding: utf-8 -*-

'tcpClient'

import socket

MAX_PACK_SIZE = 1024
DEST_IP, DEST_PORT= 'localhost', 8888

def tcpClient():   
    cliSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #create socket
    cliSock.connect((DEST_IP, DEST_PORT)) #connect  
    cliSock.send("hello sever!") #send   
    dat = cliSock.recv(MAX_PACK_SIZE)  #size
    print dat
    
    cliSock.close()

if __name__ == "__main__":   
    tcpClient()  
{% endhighlight %}

---
---

**2.UDP**

- `sever`(1)创建一个socket(2)公开所绑定的端口和地址(3)阻赛等待接收。
- `client`(1)创建一个socket(2)指定地址端口发送。
- 服务器和客户端互发消息，由于是顺序执行，所以的用到多线程来编程。
  {% highlight python %}
# -*- coding: utf-8 -*-

'udpServer'

import socket

IP, PORT = 'localhost', 8888
MAX_PACK_SIZE = 1024

def udpServer():   
    srvSock = socket.socket( socket.AF_INET, socket.SOCK_DGRAM) #socket   
    srvSock.bind((IP, PORT))  #bind ip and port
    state = 0
    while True:  
    
      dat, addr = srvSock.recvfrom(MAX_PACK_SIZE)  #backup and receive from client
      print "received:", dat, "from", addr
      dat = ""

if __name__ == "__main__":   
    udpServer()
    srvSock.close()  
{% endhighlight %}

---
{% highlight python %}
# -*- coding: utf-8 -*-

'udpClient'

import socket

MAX_PACK_SIZE = 1024
DEST_IP, DEST_PORT= 'localhost', 8888

def udpClient():   

	cliSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #create socket
	cliSock.sendto("hello sever!", (DEST_IP, DEST_PORT)) #send   
	cliSock.close()


if __name__ == "__main__":   
    udpClient()  

{% endhighlight %}
