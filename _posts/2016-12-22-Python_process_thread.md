---
layout: post
title: 进程和线程
tags:  [语言编程]
excerpt: "以Python为例，对进程和线程简要说明                                                            "
---

以Python为例，对进程和线程简要说明。

---

**(1)对于操作系统来说，一个任务就是一个进程（Process），比如打开一个浏览器就是启动一个浏览器进程，打开一个记事本就启动了一个记事本进程，打开两个记事本就启动了两个记事本进程，打开一个Word就启动了一个Word进程。**

**(2)有些进程还不止同时干一件事，比如Word，它可以同时进行打字、拼写检查、打印等事情。在一个进程内部，要同时干多件事，就需要同时运行多个“子任务”，我们把进程内的这些“子任务”称为线程（Thread）。**

**(3)进程中相同变量是不共享的，线程中相同变量是共享的，要确保balance计算正确则要给线程创建一个锁** threading.Lock 。

**(4)线程中共享变量的例子：**

{% highlight python %}

import time, threading

# 假定这是你的银行存款:
balance = 0

def change_it(n):
    # 先存后取，结果应该为0:
    global balance
    balance = balance + n
    balance = balance - n

def run_thread(n):
    for i in range(100000):
        change_it(n)

t1 = threading.Thread(target=run_thread, args=(5,))
t2 = threading.Thread(target=run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)

{% endhighlight %}

- 我们定义了一个共享变量balance，初始值为0，并且启动两个线程，先存后取，理论上结果应该为0，但是，由于线程的调度是由操作系统决定的，当t1、t2交替执行时，只要循环次数足够多，balance的结果就不一定是0了。原因是因为高级语言的一条语句在CPU执行时是若干条语句，即使一个简单的计算： balance = balance + n 也分两步：

1.计算balance + n，存入临时变量中；
2.将临时变量的值赋给balance。

**(5)多进程稳定性更好，一般进程之间不会有啥影响，只有主进程崩溃了全进程才会崩溃掉，缺点就是开销大；而多线程虽然速度快了一点，但是一个线程崩溃，所有线程崩溃掉。**
**(6)计算密集型消耗cpu，采用C语言编写最好；IO密集型(涉及磁盘和网络)采用Python最好，而C语言最差。**
**(7)在线程和进程中我们优选porcess，由于process较为稳定且能分布到多台机器上。**
