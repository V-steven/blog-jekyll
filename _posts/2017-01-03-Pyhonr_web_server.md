---
layout: post
title: Python入门Web server
tags:  [语言编程]
excerpt: "Python入门Web server,涉及到了迭代和并发处理请求的服务器、适用不同框架的服务器和一些相关的简单概念。 "
---

**How to understand the web server?**

- 位于物理服务器上的网络服务器，服务器接收到客户端的请求，就会给客户端返回一个响应，之间采用HTTP通信。
  <img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/web/1.png" style="width:500px">


**1.一个简单的WEB SERVER,运行以下程序，浏览器中输入**URL http://localhost:8888/hello

{% highlight Python %}

# -*- coding: UTF-8 -*-

import socket
HOST, PORT = '', 8888

listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #socket监听
listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listen_socket.bind((HOST, PORT))
listen_socket.listen(1)

print "Serving HTTP on port %s...." % PORT

while True:
    client_connection, client_address = listen_socket.accept()
    request = client_connection,recv(1024)
    print request
    
    http_response = """
HTTP/1.1 200 OK #响应状态行TTP/1.1 200 OK包含了HTTP版本，HTTP状态码和HTTP状态码理由短语OK

Hello,World!    #http响应body
"""
    client_connection.sendall(http_response)
    client_connection.close()

{% endhighlight %}

**2.什么叫URL?URL表示连接WEB服务器地址和你要连接的服务器上的页面路径。**

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/web/2.jpg" style="width:500px">

**3.不同python web框架共用服务器。**

- 往往不同的python web(Django,Flask,Pyramid等)框架会影响服务器。怎样才能使用一个服务器(不修改)适应不同的框架？那就使用到了WSGI。
- 下面是WSGI服务器代码实现，建立webServer.py文件并保存。
  {% highlight Python %}

# -*- coding: UTF-8 -*-

import socket
import StringIO
import sys

class WSGIServer(object):

    address_family = socket.AF_INET
    socket_type = socket.SOCK_STREAM
    request_queue_size = 1
    
    def __init__(self, server_address):
        # Create a listening socket
        self.listen_socket = listen_socket = socket.socket(
            self.address_family,
            self.socket_type
        )
        # Allow to reuse the same address
        listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind
        listen_socket.bind(server_address)
        # Activate
        listen_socket.listen(self.request_queue_size)
        # Get server host name and port
        host, port = self.listen_socket.getsockname()[:2]
        self.server_name = socket.getfqdn(host)
        self.server_port = port
        # Return headers set by Web framework/Web application
        self.headers_set = []
    
    def set_app(self, application):
        self.application = application
    
    def serve_forever(self):
        listen_socket = self.listen_socket
        while True:
            # New client connection
            self.client_connection, client_address = listen_socket.accept()
            # Handle one request and close the client connection. Then
            # loop over to wait for another client connection
            self.handle_one_request()
    
    def handle_one_request(self):
        self.request_data = request_data = self.client_connection.recv(1024)
        # Print formatted request data a la 'curl -v'
        print(''.join(
            '&lt; {line}n'.format(line=line)
            for line in request_data.splitlines()
        ))
    
        self.parse_request(request_data)
    
        # Construct environment dictionary using request data
        env = self.get_environ()
    
        # It's time to call our application callable and get
        # back a result that will become HTTP response body
        result = self.application(env, self.start_response)
    
        # Construct a response and send it back to the client
        self.finish_response(result)
    
    def parse_request(self, text):
        request_line = text.splitlines()[0]
        request_line = request_line.rstrip('rn')
        # Break down the request line into components
        (self.request_method,  # GET
         self.path,            # /hello
         self.request_version  # HTTP/1.1
         ) = request_line.split()
    
    def get_environ(self):
        env = {}
        # The following code snippet does not follow PEP8 conventions
        # but it's formatted the way it is for demonstration purposes
        # to emphasize the required variables and their values
        #
        # Required WSGI variables
        env['wsgi.version']      = (1, 0)
        env['wsgi.url_scheme']   = 'http'
        env['wsgi.input']        = StringIO.StringIO(self.request_data)
        env['wsgi.errors']       = sys.stderr
        env['wsgi.multithread']  = False
        env['wsgi.multiprocess'] = False
        env['wsgi.run_once']     = False
        # Required CGI variables
        env['REQUEST_METHOD']    = self.request_method    # GET
        env['PATH_INFO']         = self.path              # /hello
        env['SERVER_NAME']       = self.server_name       # localhost
        env['SERVER_PORT']       = str(self.server_port)  # 8888
        return env
    
    def start_response(self, status, response_headers, exc_info=None):
        # Add necessary server headers
        server_headers = [
            ('Date', 'Tue, 31 Mar 2015 12:54:48 GMT'),
            ('Server', 'WSGIServer 0.2'),
        ]
        self.headers_set = [status, response_headers + server_headers]
        # To adhere to WSGI specification the start_response must return
        # a 'write' callable. We simplicity's sake we'll ignore that detail
        # for now.
        # return self.finish_response
    
    def finish_response(self, result):
        try:
            status, response_headers = self.headers_set
            response = 'HTTP/1.1 {status}rn'.format(status=status)
            for header in response_headers:
                response += '{0}: {1}rn'.format(*header)
            response += 'rn'
            for data in result:
                response += data
            # Print formatted response data a la 'curl -v'
            print(''.join(
                '&gt; {line}n'.format(line=line)
                for line in response.splitlines()
            ))
            self.client_connection.sendall(response)
        finally:
            self.client_connection.close()

SERVER_ADDRESS = (HOST, PORT) = '', 8888

def make_server(server_address, application):
    server = WSGIServer(server_address)
    server.set_app(application)
    return server

if __name__ == '__main__':
    if len(sys.argv) &lt; 2:
        sys.exit('Provide a WSGI application object as module:callable')
    app_path = sys.argv[1]
    module, application = app_path.split(':')
    module = __import__(module)
    application = getattr(module, application)
    httpd = make_server(SERVER_ADDRESS, application)
    print('WSGIServer: Serving HTTP on port {port} ...n'.format(port=PORT))
    httpd.serve_forever()

{% endhighlight %}

- 安装Pyramid，Flask，Django等框架。
- 以Pyramid框架作为例子，建立pyramidApp.py并保存在服务器代码相同目录,运行。

{% highlight Python %}

# -*- coding: UTF-8 -*-

from pyramid.config import Configurator
from pyramid.response import Response

def hello_world(request):
    return Response(
        'Hello world from Pyramid!n',
        content_type='text/plain',
    )

config = Configurator()
config.add_route('hello', '/hello')
config.add_view(hello_world, route_name='hello')
app = config.make_wsgi_app()

{% endhighlight %}

**4.TCP Socket理解。**

- Socket编程基本就是create，listen，accept，connect，read和write等几个基本的操作。
- TCP/IP仅仅只是一个协议栈，就像操作系统的运行机制一样，操作系统必须提供对外的操作接口。就像操作系统会提供标准的编程接口，比如Win32编程接口一样，TCP/IP也必须对外提供编程接口，这就是Socket编程接口(API)。
- Socket需要与端口绑定，端口对应的是一个节点(计算机)中的一个应用，端口绑定后其他应用不能使用。
- TCP的Socket对是一个4元组，标识着TCP连接的两个终端：本地IP地址、本地端口、远程IP地址、远程端口。一个Socket对唯一地标识着网络上的TCP连接。标识着每个终端的两个值，IP地址和端口号。
  <img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/web/3.png" style="width:500px">

**5.服务器同时处理多个请求**

- 如前都是采用的迭代方式来处理客户端的请求，即每次服务器只处理一个请求，客户端排队接收相应。
- 验证迭代处理请求，每次处理请求后让服务器休眠60s,连续向服务器发送两次请求，则之间会相隔60s时间。

{% highlight Python %}

import socket
import time

SERVER_ADDRESS = (HOST, PORT) = '', 8888
REQUEST_QUEUE_SIZE = 5

def handle_request(client_connection):
    request = client_connection.recv(1024)
    print(request.decode())
    http_response = b"""
HTTP/1.1 200 OK

Hello, World!
"""
    client_connection.sendall(http_response)
    time.sleep(60)  # sleep and block the process for 60 seconds

def serve_forever():
    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_socket.bind(SERVER_ADDRESS)
    listen_socket.listen(REQUEST_QUEUE_SIZE)
    print('Serving HTTP on port {port} ...'.format(port=PORT))
    
    while True:
        client_connection, client_address = listen_socket.accept()
        handle_request(client_connection)
        client_connection.close()

if __name__ == '__main__':
    serve_forever()

{% endhighlight %}

- 在Unix下，利用fork()多进程的方式使服务器同时处理多个请求。
- 当一个进程fork了一个新进程时，它就变成了那个新fork产生的子进程的父进程。
- 在调用fork后，父进程和子进程共享相同的文件描述符。
- 服务器父进程的角色是：现在它干的所有活就是接受一个新连接，fork一个子进来来处理这个请求，然后循环接受新连接。

{% highlight python %}

import os
import socket
import time

SERVER_ADDRESS = (HOST, PORT) = '', 8888
REQUEST_QUEUE_SIZE = 5

def handle_request(client_connection):
    request = client_connection.recv(1024)
    print(
        'Child PID: {pid}. Parent PID {ppid}'.format(
            pid=os.getpid(),
            ppid=os.getppid(),
        )
    )
    print(request.decode())
    http_response = b"""
HTTP/1.1 200 OK

Hello, World!
"""
    client_connection.sendall(http_response)
    time.sleep(60)

def serve_forever():
    listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listen_socket.bind(SERVER_ADDRESS)
    listen_socket.listen(REQUEST_QUEUE_SIZE)
    print('Serving HTTP on port {port} ...'.format(port=PORT))
    print('Parent PID (PPID): {pid}n'.format(pid=os.getpid()))
    
    while True:
        client_connection, client_address = listen_socket.accept()
        pid = os.fork()
        if pid == 0:  # child
            listen_socket.close()  # close child copy
            handle_request(client_connection)
            client_connection.close()
            os._exit(0)  # child exits here
        else:  # parent
            client_connection.close()  # close parent copy and loop over

if __name__ == '__main__':
    serve_forever()

{% endhighlight %}

---
---
