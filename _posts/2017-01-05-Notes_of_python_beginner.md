---
layout: post
title: Notes of python beginner
tags:  [语言编程]
excerpt: "不同的编程语言都有它独自的特点，python作为主攻编写应用程序的高级编程语言，它开发效率高和有完善的代码库，'优雅'.'明确'.'简单'的特点吸引了我，所以打算好好学习python，以下是初学Python的笔记整理。"
---
不同的编程语言都有它独自的特点，python作为主攻编写应用程序的高级编程语言，它开发效率高和有完善的代码库，优雅.明确.简单的特点吸引了我，所以打算好好学习python，以下是初学Python的笔记整理。

---

### 一张Python简要格式和语法解说图，适合初学者看

<img src="http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/Python/python_beginner_cn.png" style="width:1200px">


**1.list/tuple**

- list是有序的集合，可以随意删除和添加其中的元素。
- tuple初始化就不能修改，没有append()，insert()方法。

{% highlight python %}

classmates = ['Michael', 'Bob', 'Tracy'] #list  
len(classmates)   
classmates[-1] #获取最后一个元素，以此推[-1]equal[2],[-2]equal[1],[-3]euqal[0]
classmates.append('Adam') #末尾添加元素   
classmates.insert(1, 'Jack') #指定位置添加元素    
classmates.pop(1) #删除之指定位置的元素，为空时为末尾元素   
s = ['python', 'java', ['asp', 'php'], 'scheme'] #list中可以用list作为元素   
----------------------------------------------------------------------------------------
classmates = ('Michael', 'Bob', 'Tracy') #tuple初始化的中括号和圆括号的区别
t = ('a', 'b', ['A', 'B']) #元组中的元素可以是利列表
t[2][0] = 'X' #列表中的元素是可以改变的

{% endhighlight %}

---

**2.input/for/while**

- `input()`返回的是字符串，数值输入时须通过`int()`将其转换。

{% highlight python %}

age = int(input('Input your age: '))

{% endhighlight %}

- for迭代list和tuple元素。
- while用法没什么特别。

{% highlight python %}

names = ['Michael', 'Bob', 'Tracy']
for name in names:
    print(name)

for ch in 'ABC':
  print(ch)
----------------------------------------------------------------------------------------
sum = 0
n = 1
while n <= 100:
    sum = sum + n
    n = n + 1
{% endhighlight %}

---

**3.dict/set**

- dict字典采用键值存储，和list相比查找快但更耗内存空间换取时间。  
- set只存键，用list表示，无重复的键，两个set可以交集和并集运算。

{% highlight python %}

d = {'Michael': 95, 'Bob': 75, 'Tracy': 85}
print d['Michael']
----------------------------------------------------------------------------------------
s1 = set([1, 2, 3])
s2 = set([2, 3, 4])
s1 & s2 #{2, 3}
s1 | s2 #{1, 2, 3, 4}

{% endhighlight %}

---

**4.函数/匿名函数**

- 函数可以返回多个值tuple。
- 参数默认时，则传入的参数怎可以省略默认的。
- 参数变量前加*，则可以传入list和tuple。
- 参数变量前加**,则可以传入dict。
- 匿名函数lambda。

{% highlight python %}

def move(x, y, step, angle=0):
    nx = x + step * math.cos(angle)
    ny = y - step * math.sin(angle)
    return nx, ny
n = my_abs(-20)
print(n) #多值返回(151.96152422706632, 70.0)
----------------------------------------------------------------------------------------
def calc(*numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum
nums = [1, 2, 3]
calc(*nums) #传入list列表，简单应用calc(1,2,3)
----------------------------------------------------------------------------------------
def person(name, age, **kw):
    if 'city' in kw: #在dict字典中匹配,有city参数
        pass
    if 'job' in kw:  #有job参数
        pass
    print('name:', name, 'age:', age, 'other:', kw)

person('Jack', 24, city='Beijing', addr='Chaoyang', zipcode=123456) #传入参数不受限制
----------------------------------------------------------------------------------------
f = lambda x: x * x #等同于def f(x):return x * x,冒号前面为参数
f(5)

{% endhighlight %}

---

**5.递归**

- 递归，函数内部调用本身，容易发生栈溢出。
- 尾递归，调用本身时不含其他表达式，不会出现栈溢出。

{% highlight python %}

def fact(n): # 普通递归函数
    if n==1:
        return 1
    return n * fact(n - 1)
print fact(5) # 120

# 如下是调用过程
===> fact(5)
===> 5 * fact(4)
===> 5 * (4 * fact(3))
===> 5 * (4 * (3 * fact(2)))
===> 5 * (4 * (3 * (2 * fact(1))))
===> 5 * (4 * (3 * (2 * 1)))
===> 5 * (4 * (3 * 2))
===> 5 * (4 * 6)
===> 5 * 24
===> 120
----------------------------------------------------------------------------------------
def fact_iter(num, product): #采用尾递归
    if num == 1:
        return product
    return fact_iter(num - 1, num * product) #不含其他表达式，只有函数本身
print fact_iter(5, 1) # 120

#如下是调用过程
===> fact_iter(5, 1)
===> fact_iter(4, 5)
===> fact_iter(3, 20)
===> fact_iter(2, 60)
===> fact_iter(1, 120)
===> 120

{% endhighlight %}

---

**6.切片**
{% highlight python %}

 L = list(range(100))
 L[:10] #前十个元素
 L[-10:] #后十个元素
 L[:10:2] #前十个每隔两个取一个

{% endhighlight %}

---

**7.列表生成式**
{% highlight python %}

print([x * x for x in range(1, 11)]) #生成列表

L = ['Hello', 'World', 'IBM', 'Apple'] #使用low()方法将大写转换为小写，若列表内有非字符串则报错
print([s.lower() for s in L])

{% endhighlight %}

---

**8.生成器**

- 遇到yield(`中断`)语句返回，再次执行时从上次返回的yield语句处继续执行。
  {% highlight python %}

def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b #和print的区别，print不返回可用值，复用性交叉，而yield返回可用值构成list
        a, b = b, a + b #先计算右边(b,a+b)，在计算等值(a=b,b=a+b)
        n = n + 1
    return

for x in fib(10): #用next()或者for迭代
    print x

{% endhighlight %}

---

**9.map/reduce/filter/sorted**

- `map`将传入的函数依次作用到序列的每个元素。
- `reduce`把结果继续和序列的下一个元素做累积计算。
- `filter`把传入的函数依次作用于每个元素，然后根据返回值是True还是False决定保留还是丢弃该元素。
- `sorted`对list序列进行排序，数值和字符串。
  {% highlight python %}

def func(x):
  return x * x
r = map(func, [1, 2, 3, 4, 5, 6, 7, 8, 9])
list(r) #[1, 4, 9, 16, 25, 36, 49, 64, 81]
----------------------------------------------------------------------------------------
from functools import reduce
def add(x, y):
  return x + y
reduce(add, [1, 3, 5, 7, 9]) #结果25 其效果为reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)
----------------------------------------------------------------------------------------
def _odd_iter(): #奇数序列
    n = 1
    while True:
        n = n + 2
        yield n

def _not_divisible(n): #质数
    return lambda x: x % n > 0

def primes():
    yield 2 #返回质数2
    it = _odd_iter() #初始化
    while True:
        n = next(it) #列出质数
        yield n #返回质数序列
        it = filter(_not_divisible(n), it) #用filter从奇数中选出质数

if __name__ == '__main__': #求质数
    for n in primes():
    if n < 10:
        print(n)
    else:
        break
# BUT MemoryError
{% endhighlight %}

---

**10.装饰器**

- 装饰器是一个很著名的设计模式，经常被用于有切面需求的场景，较为经典的有插入日志、性能测试、事务处理等。装饰器是解决这类问题的绝佳设计，有了装饰器，我们就可以抽离出大量函数中与函数功能本身无关的雷同代码并继续重用。概括的讲，装饰器的作用就是为已经存在的对象添加额外的功能。
- 首先必须明白在Python中函数也是被视为对象。
- @timeit与foo = timeit(foo)完全等价。

{% highlight python %}
def timeit(func):
    def wrapper(): #封装
        start = time.clock()
        func()
        end =time.clock()
        print 'used:', end - start
    return wrapper

@timeit #等同于timeit(foo)
def foo():
    print 'in foo()'

foo()
{% endhighlight %}

---

**11.面向对象**

- __init__方法的第一个参数永远是self，同时绑定name和score，有此方法不能为空参数。
- 内部属性不被外部访问，可以把属性的名称前加上两个下划线__。
- __init__和__del__类似于c++里的构造和析构。
- 继承和多态，子类和父类有同名方法，将隐藏父类的方法。
- 通过type()函数创建出Hello类，而无需通过class Hello(object)...的定义。
  {% highlight python %}
  class Student(object): #继承object类
    def __init__(self, name, score): #__init__方法的第一个参数永远是self，绑定name和score，方法不能为空参数
        self.name = name
        self.score = score

    def print_score(self): #定义方法时，同样第一个参数为self
        print('%s: %s' % (self.name, self.score))
  bart = Student('Bart Simpson', 59)
  bart.print_score()
----------------------------------------------------------------------------------------
class Student(object):
    def __init__(self, name, score): #外部不能访问name和score
        self.__name = name
        self.__score = score
    
    def get_name(self): #通过此方法，外部可以访问值
          return self.__name
    
    def print_score(self):
        print('%s: %s' % (self.__name, self.__score))
bart = Student('Bart Simpson', 98)
bart.__name #Error
bart.get_name()
----------------------------------------------------------------------------------------
class Animal(object):
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def showAnimal(self):
        print ("Animal\'s name:%s Animal\'s age:%s" %(self.name,self.age))
class Dog(Animal):
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def show(self):
        print ("Dog\'s name:%s Dog\'s age:%s" %(self.name,self.age))
class LittleDog(Dog):
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def show(self):
        print ('LittleDog\'s name:%s LittleDog\'s age:%s' %(self.name,self.age))

if __name__ == '__main__':
    littleDog = LittleDog('Huang', 30)
    littleDog.showAnimal() #调用父类的方法
    littleDog.show() #子类和父类有同名方法，将隐藏父类的方法。
----------------------------------------------------------------------------------------
def fn(self, name='world'): # 先定义函数
    print('Hello, %s.' % name)

Hello = type('Hello', (object,), dict(hello=fn)) # 创建Hello class

h = Hello()
print('call h.hello():')
h.hello()
print('type(Hello) =', type(Hello))
print('type(h) =', type(h))
{% endhighlight %}

---

**12.错误处理**

- 检查错误代码，则用try来运行这段代码，如果执行出错，依次执行except>finally>end。
- 用logging打印出错误堆栈。
- 错误通过raise语句抛出去。
  {% highlight python %}

try:
    print('try...')
    r = 10 / 0
    print('result:', r)
except ZeroDivisionError as e:
    print('except:', e)
finally:
    print('finally...')

print('END')
----------------------------------------------------------------------------------------
import logging

def foo(s):
    return 10 / int(s)

def bar(s):
    return foo(s) * 2

def main():
    try:
        bar('0')
    except Exception as e:
        logging.exception(e)

main()
print('END')
----------------------------------------------------------------------------------------
class FooError(ValueError):
    pass

def foo(s):
    n = int(s)
    if n==0:
        raise FooError('invalid value: %s' % s)
    return 10 / n

foo('0')
{% endhighlight %}

---

**13.IO编程**

- 同步IO：将数据写入磁盘时，需要等待磁盘写完了数据cpu在接着执行之后的事情。
- 异步IO：将数据写入磁盘是，cpu不需要等待磁盘写完数据，可以执行其他的事儿。
- 读文件时，若文件不在报错，写文件时，没文件则创建文件。
- 文件操作后要close,为了防止数据丢失，用with语句。
- 在内存进行读写，使用StringIO和BytesIO，

{% highlight python %}
from datetime import datetime

with open('test.txt', 'w') as f:
    f.write('今天是 ')
    f.write(datetime.now().strftime('%Y-%m-%d'))

with open('test.txt', 'r') as f:
    s = f.read()

with open('test.txt', 'rb') as f:
    s = f.read()
{% endhighlight %}

---

**14.JSON**

- 序列化为JSON格式。
  {% highlight python %}

import json

d = dict(name='Bob', age=20, score=88)
data = json.dumps(d)
print('JSON Data is a str:', data)
reborn = json.loads(data) #  序列化为JSON格式
print(reborn)

class Student(object):

    def __init__(self, name, age, score):
        self.name = name
        self.age = age
        self.score = score
    
    def __str__(self): #要将Student类序列化为JSON，必须由此函数转换，执行print类时首先调用__str__方法
        return 'Student object (%s, %s, %s)' % (self.name, self.age, self.score)

s = Student('Bob', 20, 88)
std_data = json.dumps(s, default=lambda obj: obj.__dict__)
print('Dump Student:', std_data)
rebuild = json.loads(std_data, object_hook=lambda d: Student(d['name'], d['age'], d['score']))
print(rebuild)
{% endhighlight %}

---

**15.base64**
{% highlight python %}
import base64

s = base64.b64encode('在Python中使用BASE 64编码')
print(s)
d = base64.b64decode(s) #编码可逆
print(d)

s = base64.urlsafe_b64encode('在Python中使用BASE 64编码') #urlsafe将+和/分别变成-和_
print(s)
d = base64.urlsafe_b64decode(s)
print(d)
{% endhighlight %}

---

**16.hashlib**

- hashlib提供了MD5(128 bit)、SHA1(160 bit)摘要算法,用于对数据的加密。
- 更安全的做法，对加密数据进行加盐salt，方式password + the-Salt。

{% highlight python %}
import hashlib
md5 = hashlib.md5()
md5.update('how to use md5'.encode('utf-8'))
print(md5.hexdigest())
----------------------------------------------------------------------------------------
import hashlib
sha1 = hashlib.sha1()
sha1.update('how to use sha1'.encode('utf-8'))
print(sha1.hexdigest())

{% endhighlight %}

---

**17.Tkinter**

- 在python2.7环境下使用Tkinter进行GUI编程。
  {% highlight python %}
# -*- coding: utf-8 -*-

'a hello world GUI example.'

from Tkinter import *
import tkMessageBox

class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack() #call method function
        self.createWidgets()
    
    def createWidgets(self):
        self.nameInput = Entry(self) #create input widget
        self.nameInput.pack() #add to window
        self.alertButton = Button(self, text='Hello', command=self.hello) #listen and excute
        self.alertButton.pack()
    
    def hello(self):
        name = self.nameInput.get() or 'world' #input label
        tkMessageBox.showinfo('Message', 'Hello, %s' % name) #message box

app = Application() #title of window
app.master.title('Hello World')
app.mainloop()

{% endhighlight %}
