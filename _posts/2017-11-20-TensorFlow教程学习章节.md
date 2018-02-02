---
layout: post
title: TensorFlow教程学习章节
tags: [深度学习]
excerpt: "TensorFlow概述、图与会话、图的边与节点、常量、变量、占位符、名字与作用域、文件IO与模型存取、队列与线程、数据集文件操作、TensorBoard基础用法"

---



# **TensorFlow基础教程**

基于TensorFlow最新（1.3）版本的中文基础教程。适用于有一定Python基础的学生入门。

**目录：**

* TensorFlow概述
* 图与会话
* 图的边与节点
* 常量、变量、占位符
* 名字与作用域
* 文件IO与模型存取
* 队列与线程
* 数据集文件操作
* TensorBoard基础用法



---


# 一.Tensorflow概述

### 什么是Tensorflow

Tensorflow是由Google Brain Team开发的使用数据流图进行数值计算的开源机器学习库。Tensorflow的一大亮点是支持异构设备分布式计算(heterogeneous distributed computing)。这里的异构设备是指使用CPU、GPU等计算设备进行有效地协同合作。

*Google Brain Team与DeepMind是独立运行相互合作的关系。*

Tensorflow拥有众多的用户，除了Alphabet内部使用外，ARM、Uber、Twitter、京东、小米等众多企业均使用Tensorflow作为机器学习的工具。

![]( http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/3.png)



常见机器学习库包括Tensorflow、MXNet、Torch、Theano、Caffe、CNTK、scikit-learn等。

|                  库                  | 维护人员或机构                         |             支持语言              |              支持操作系统              |
| :---------------------------------: | :------------------------------ | :---------------------------: | :------------------------------: |
|             Tensorflow              | google                          |         Python、C++、Go         | Linux、mac os、Android、iOS、Windows |
|                MXNet                | 分布式机器学习社区(DMLC)                 | Python、Scala、R、Julia、C++、Perl | Linux、mac os、Android、iOS、Windows |
|                Torch                | Ronan Collobert等人               |         Lua、LuaJIT、C          | Linux、mac os、Android、iOS、Windows |
|               Theano                | 蒙特利尔大学( Université de Montréal) |            Python             |       Linux、mac os、Winodws       |
| Computational Network Toolkit(CNTK) | 微软研究院                           |    Python、C++、BrainScript     |          Linux、Windows           |
|                Caffe                | 加州大学伯克利分校视觉与学习中心                |       Python、C++、MATLAB       |       Linux、mac os、Windows       |
|            PaddlePaddle             | 百度                              |          Python、C++           |           Linux、mac os           |



各个框架对比https://github.com/zer0n/deepframeworks

![框架热度对比]( http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/2.png)

### Tensor

Tensor是张量的意思，原本在物理学中用来描述大于等于2维的量进行量纲分析的工具。我们早已熟知如何处理0维的量（纯量）、1维的量（向量）、2维的量（矩阵）。对于高维的数据，我们也需要一个工具来表述，这个工具正是张量。

张量类似于编程语言中的多维数组（或列表）。广义的张量包括了常量、向量、矩阵以及高维数据。在处理机器学习问题时，经常会遇到大规模样本与大规模计算的情况，这时候往往需要用到张量来进行计算。Tensorflow中张量是最重要与基础的概念。

### 编程模式

编程模式通常分为**命令式编程（imperative style programs）**和**符号式编程（symbolic style programs）**。命令式编程，直接执行逻辑语句完成相应任务，容易理解和调试；符号式编程涉及较多的嵌入和优化，很多任务中的逻辑需要使用图进行表示，并在其他语言环境中执行完成，不容易理解和调试，但运行速度有同比提升。

命令式编程较为常见，例如直接使用C++、Python进行编程。例如下面的代码：

~~~python
import numpy as np
a = np.ones([10,])
b = np.ones([10,]) * 5
c = a + b
~~~

当程序执行到最后一句时，a、b、c三个变量有了值。程序执行的是真正的计算。

符号式编程不太一样，仍然是完成上述功能，使用符号式编程的写法如下（伪代码）：

~~~python
a = Ones_Variables('A', shape=[10,])
b = Mul(Ones_Variables('B', shape=[10,]), 5)
c = Add(a, b)

# 计算
Run(c)
~~~

上述代码执行到c=Add(a, b)时，并不会真正的执行加法运算，同样的a、b也并没有对应的数值，a、b、c均是一个符号，符号定义了执行运算的结构，我们称之为**计算图**，计算图没有执行真正的运算。当执行Run(c)时，计算图开始真正的执行计算，计算的环境通常不是当前的语音环境，而是C++等效率更高的语言环境。

机器学习库中，Tensorflow、theano使用了符号式编程；Torch使用了命令式编程；caffe、mxnet采用了两种编程模式混合的方式。

### 数据流图

当我们使用计算图来表示计算过程时，事实上可以看做是一个推断过程。在推断时，我们输入一些数据，并使用符号来表示各种计算过程，最终得到一个或多个推断结果。所以使用计算图可以在一定程度上对计算结果进行预测。

计算图在推断的过程中也是数据流转的过程，所以我们也可以称之为**数据流图**。举个例子，假如我们计算$(a+b)*(b+1)$的值，那么我们画出其数据流图，如下：

![]( http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/tree-def.png)

输入a与b，通过不同通路进行计算并传入下一个节点。这一过程就是数据流动的过程。有了数据流图，我们还可以进行更多的操作，例如自动求微分等，在此不做赘述。

### Tensorflow高层库

Tensorflow本质上是数值计算库，在数值处理与计算方面比较方便、灵活。虽然Tensorflow为机器学习尤其是深度学习提供了很多便捷的API，但在构建算法模型时，仍然较为复杂。为此Tensorflow官方以及众多第三方机构与个人开发了很多的使用简便的高层库，这些库与Tensorflow完全兼容，但可以极大简化模型构建、训练、部署等操作。其中较为常用工具包与高层库为：

1. TF Learn(tf.contrib.learn)：类似于scikit-learn的使用极少代码量即可构建机器学习算法的工具包。
2. TF Slim(tf.contrib.slim)：一个轻量级的用于定义、训练、评估深度学习算法的Tensorflow工具包。
3. 高级API：Keras，TFLearn，Pretty Tensor

### Tensorflow的发展

2015年11月9日，Tensorflow的0.5的版本发布并开源。起初Tensorflow的运行效率低下，不支持分布式、异构设备，并不被看好。2016年4月，经过不到半年的时间发布了0.8版本，开始支持分布式、多GPU运算，2016年6月，0.9的版本改进了对移动设备的支持，到此时，Tensorflow已经成为了为数不多的支持分布式、异构设备的开源机器学习库，并极大的改善了运算效率问题，成为运算效率最高的机器学习算法库之一。2017年2月，Tensorflow的1.0正式版发布，增加了专用的编译器XLA、调试工具Debugger和tf.transform用来做数据预处理，并开创性的设计了Tensorflow Fold用于弥补符号编程在数据预处理时的缺陷，成为了行业领先的机器学习库。到现在Tensorflow已经成为了众多企业、机构中最常用的机器学习库。

## Tensorflow能干什么？

1. 设计机器算法。
2. 训练机器学习算法。
3. 部署算法到多种设备上。
4. 很适合做深度学习。

---



# 二.图与会话


Tensorflow使用**数据流图(Data Flow Graphs)**来定义计算流程。数据流图在定义阶段并不会执行出计算结果。数据流图使得计算的定义与执行分离开来。Tensorflow构建的数据流图是有向无环图。而图的执行需要在**会话(Session)**中完成。

## 1. 数据流图

我们尝试构建一个简单的图，这个图描述了两个数的加法运算，如下：

~~~python
import tensorflow as tf

a = tf.add(3, 5)
~~~

使用TensorBoard可视化图：

<img src=' http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/graph_add.png' width='300px'>

此处的x、y是被TensorBoard自动命名的，分别代表3与5这两个量。这里需要注意的是数据流图含了**边（edge）**和**节点（node）**，正好与Tensorflow中的tensor与flow对应。tensor代表了数据流图中的边，而flow这个动作代表了数据流图中的节点。节点也被称之为**操作（operation, op）**，一个 op 获得 0 个或多个 `Tensor`, 执行计算, 产生 0 个或多个 `Tensor`。 每个 Tensor 是一个类型化的多维数组 。关于节点与边，我们会在后续的内容当中详细描述。

这时候如果我们直接输出a，会得到如下结果：

~~~python
print(a)

>> Tensor("Add:0", shape=(), dtype=int32)
~~~

根据我们的推断，执行加法之后，应该得到一个值，为‘8’，然而并非如此。我们得到的是一个Tensor对象的描述。之所以如此是因为我们定义的变量a代表的是一个图的定义，这个图定义了一个加法运算，但并没有执行。要想得到计算结果，必须在会话中执行。

## 2. 会话

启动会话的第一步是创建一个session对象。会话提供在图中执行op的一些列方法。一般的模式是，建立会话，在会话中添加图，然后执行。建立会话可以有两种方式，一种是`tf.Session()`，通常使用这种方法创建会话；另一种是`tf.InteractiveSession()`，这种方式更多的用在iPython notebook等交互式Python环境中。

在会话中执行图：

~~~python
import tensorflow as tf

# 创建图
a = tf.add(3, 5)
# 创建会话
sess = tf.Session()
# 执行图
res = sess.run(a)  # print(res) 可以得到 8
# 关闭会话
sess.close()
~~~

为了简便，我们可以使用上下文管理器来创建Session，所以也可以写成：

~~~python
a = tf.add(3, 5)
with tf.Session() as sess:
	print(sess.run(a))
~~~



### 2.1 feed与fetch

在调用Session对象的run()方法来执行图时，传入一些张量，这一过程叫做**填充（feed）**，返回的结果类型根据输入的类型而定，这个过程叫**取回（fetch）**。

`tf.Session.run()`方法可以运行并评估传入的张量：

~~~python
# 运行fetches中的所有张量
run(fetches, feed_dict=None, options=None, run_metadata=None)
~~~

除了使用`tf.Session.run()`以外，在sess持有的上下文中还可以使用`eval()`方法。

~~~python
a = tf.add(3, 5)
with tf.Session() as sess:
	print(a.eval())
~~~

这里需要注意，`a.eval()`必须在sess持有的上下文中执行才可行。有时候，在交互式的Python环境中，上下文管理器使用不方便，这时候我们可以使用交互式会话来代替会话，这样`eval()`方法便可以随时使用：

~~~python
a = tf.add(3, 5)
sess = tf.InteractiveSession()
print(a.eval())
sess.close()
~~~

两种会话的区别就是是否支持直接使用`eval()`。



### 2.2 节点依赖

通常一个图会有较多的边与节点，这时候在会话中执行图时，所有依赖的节点均参与计算，如：

~~~python
import tensorflow as tf

x = 2
y = 3

op1 = tf.add(x, y)
op2 = tf.multiply(x, y)
op3 = tf.pow(op2, op1)

with tf.Session() as sess:
	res = sess.run(op3)
~~~

利用TensorBoard可视化图，如下：

<img src=' http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/graph_more.png' width='450px'>

虽然`session`中运行的是节点`op3`，然而与之相关联的`op1`、`op2`也参与了运算。



## 3. 子图

图的构建可以是多种多样的，以上构建的图中数据只有一个流动方向，事实上我们可以构建多个通路的图，每一个通路可以称之为其子图。如下：

~~~python
import tensorflow as tf

x = 2
y = 3

add_op = tf.add(x, y)
mul_op = tf.multiply(x, y)

useless = tf.mul(x, add_op)
pow_op = tf.pow(add_op, mul_op)

with tf.Session() as sess:
	res = sess.run(pow_op)
~~~

利用TensorBoard可视化图：

<img src=' http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/sub_graph.png' width='450px'>

可以看到此图中有两个子图，即运行完整的图我们可以得到两个结果`useless`和`pow_op`。当我们执行上述代码时，相当于执行了一个子图`pow_op`。这里需要注意由于得到`pow_op`用不到`useless`的值，即没有依赖关系，所以`useless`这个node并没有执行，如果我们需要得到`useless`和`pow_op`两个节点的值，则需要稍微改进代码：

~~~python
import tensorflow as tf

x = 2
y = 3

add_op = tf.add(x, y)
mul_op = tf.multiply(x, y)

useless = tf.multiply(x, add_op)
pow_op = tf.pow(add_op, mul_op)

with tf.Session() as sess:
	res1, res2 = sess.run([pow_op, useless])
~~~



## 4. 多个图

一个图包含了一组op对象和张量，描述了一个计算流程。但有时候我们需要建构多个图，在不同的图中定义不同的计算，例如一个图描述了算法的训练，另一个图描述了这个算法的正常运行过程，这时候需要在不同的会话中调用不同的图。

创建图使用`tf.Graph()`。将图设置为默认图使用`tf.Graph.as_default()`，此方法会返回一个上下文管理器。如果不显示的添加一个默认图，系统会自动设置一个全局的默认图。所设置的默认图，在模块范围内定义的节点都将自动加入默认图中。

```python
# 构建一个图
g = tf.Graph()
# 将此图作为默认图
with g.as_default():
	op = tf.add(3, 5)
```

上述代码也可以写成：

```python
with tf.Graph().as_default():
	op = tf.add(3, 5)
```

`tf.Graph()`*不能用作为上下文管理器，必须调用其方法*`as_default()`。

构建多个图：

```
g1 = tf.Graph()
with g1.as_default():
	pass

g2 = tf.Graph()
with g2.as_default():
	pass
```

在我们加载了Tensorflow包时，就已经加载了一个图，这也是我们可以不创建图而直接构建边和节点的原因，这些边与节点均加入了默认图。我们可以使用`tf.get_default_graph()`方法获取到当前环境中的图，如下：

```
# 获取到了系统的默认图
g1 = tf.get_default_graph()
with g1.as_default():
	pass

g2 = tf.Graph
with g2.as_default():
	pass
```

`tf.get_default_graph()`获取到的图时当前所在图的环境。如果在上面的代码中在`g2`的图环境中执行`tf.get_default_graph()`，则得到的图是`g2`。

*在使用notebook时，由于notebook的一些特性，通常需要指定默认图，否则程序可能报错。为了书写规范，推荐尽量书写完整默认图。*

当使用了多个图时，在`session`中就需要指定当前`session`所运行的图，不指定时，为默认图。如下：

```python
# 指定当前会话处理g这个图
with tf.Session(graph=g) as sess:
	sess.run(op)
```



## 5. 分布式计算

当我们有多个计算设备时，可以将一个计算图的不同子图分别使用不同的计算设备进行运算，以提高运行速度。一般，我们不需要显示的指定使用CPU或者GPU进行计算，Tensorflow能自动检测，如果检测到GPU，Tensorflow会尽可能的利用找到的第一个GPU进行操作。如果机器上有超过一个可用的GPU，出第一个外，其它的GPU默认是不参与计算的。为了让Tensorflow使用这些GPU，我们必须明确指定使用何种设备进行计算。

TensorFlow用指定字符串 `strings` 来标识这些设备. 比如:

- `"/cpu:0"`: 机器中的 CPU
- `"/gpu:0"`: 机器中的 GPU, 如果你有一个的话.
- `"/gpu:1"`: 机器中的第二个 GPU, 以此类推...

这里需要注意的是，CPU可以支持所有类型的运行，而GPU只支持部分运算，例如矩阵乘法。下面是一个使用第3个GPU进行矩阵乘法的运算：

~~~python
with tf.device('/gpu:2'):
	a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='a')
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name='b')
  	c = tf.matmul(a, b)
~~~

为了查看节点和边被指派给哪个设备运算，可以在`session`中配置“记录设备指派”为开启，如下：

~~~python
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(c))
~~~

当指定的设备不存在时，会出现 `InvalidArgumentError` 错误提示。为了避免指定的设备不存在这种情况，可以在创建的 `session` 里把参数 `allow_soft_placement` 设置为 `True`，这时候TF会在设备不存在时自动分配设备。如下：

~~~python
config = tf.ConfigProto(
	log_device_placement=True,
	allow_soft_placement=True)
with tf.Session(config=config) as sess:
	pass
~~~



## 6. 使用图与会话的优点

对于图与会话：

* 多个图需要多个会话来运行，每个会话会尝试使用所有可用的资源。
* 不同图、不同会话之间无法直接通信。可以通过python等进行间接通信。
* 最好在一个图中有多个不连贯的子图。

使用图的优点：

* 节省计算资源。我们仅仅只需要计算用到的子图就可以了，无需完整计算整个图。
* 可以将复杂计算分成小块，进行自动微分等运算。
* 促进分布式计算，将工作分布在多个CPU、GPU等设备上。
* 很多机器学习模型可以使用有向图进行表示，与计算图一致。

---

# 三.图的边与节点

图包含了节点与边。边可以看做是流动着的数据以及相关依赖关系。而节点表示了一种操作，即对当前流动来的数据的运算。

## 1. 边（edge）

Tensorflow的边有两种连接关系：**数据依赖**和**控制依赖**。其中，实线边表示数据依赖，代表数据，即张量。虚线边表示控制依赖（control dependency），可用于控制操作的运行，这被用来确保happens-before关系，这类边上没有数据流过，但源节点必须在目的节点开始执行前完成执行。



### 1.1 数据依赖

数据依赖很容易理解，某个节点会依赖于其它节点的数据，如下所示：

~~~python
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='a')
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name='b')
c = tf.matmul(a, b)
~~~



### 1.2 控制依赖

控制依赖使用这个方法`tf.Graph.control_dependencies(control_inputs)`，返回一个可以使用上下文管理器的对象，用法如下：

~~~python
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='a')
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name='b')
c = tf.matmul(a, b)

g = tf.get_default_graph()
with g.control_dependencies([c]):
    d = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='d')
    e = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name='e')
    f = tf.matmul(d, e)
with tf.Session() as sess:
    sess.run(f)
~~~

上面的例子中，我们在会话中执行了f这个节点，可以看到与c这个节点并无任何数据依赖关系，然而f这个节点必须等待c这个节点执行完成才能够执行f。最终的结果是c先执行，f再执行。

控制依赖除了上面的写法以外还拥有简便的写法：`tf.control_dependencies(control_inputs)`。其调用默认图的`tf.Graph.control_dependencies(control_inputs)`方法。上面的写法等价于：

~~~
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='a')
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name='b')
c = tf.matmul(a, b)

with tf.control_dependencies([c]):
    d = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='d')
    e = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], name='e')
    f = tf.matmul(d, e)
with tf.Session() as sess:
    sess.run(f)
~~~

**注意**：有依赖的op必须写在`tf.control_dependencies`上下文中，否则不属于有依赖的op。**如下写法是错误的**：

~~~python
def my_fun():
    a = tf.constant(1)
    b = tf.constant(2)
    c = a + b

    d = tf.constant(3)
    e = tf.constant(4)
    f = a + b
    # 此处 f 不依赖于 c
	with tf.control_dependencies([c]):
        return f
~~~



### 1.3 张量的阶、形状、数据类型

Tensorflow数据流图中的边用于数据传输时，数据是以张量的形式传递的。张量有阶、形状和数据类型等属性。

#### Tensor的阶

在TensorFlow系统中，张量的维数来被描述为**阶**。但是张量的阶和矩阵的阶并不是同一个概念。张量的阶是张量维数的一个数量描述。比如，下面的张量（使用Python中list定义的）就是2阶.

```python
t = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

你可以认为一个二阶张量就是我们平常所说的矩阵，一阶张量可以认为是一个向量.对于一个二阶张量你可以用语句`t[i, j]`来访问其中的任何元素.而对于三阶张量你可以用`t[i, j, k]`来访问其中的任何元素。

| 阶    | 数学实例          | Python 例子                                |
| ---- | ------------- | ---------------------------------------- |
| 0    | 纯量 (或标量。只有大小) | `s = 483`                                |
| 1    | 向量(大小和方向)     | `v = [1.1, 2.2, 3.3]`                    |
| 2    | 矩阵(数据表)       | `m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]`  |
| 3    | 3阶张量 (数据立体)   | `t = [[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]]` |
| n    | n阶            | `....`                                   |

#### Tensor的形状

TensorFlow文档中使用了三种记号来方便地描述张量的维度：阶，形状以及维数.下表展示了他们之间的关系：

| 阶    | 形状               | 维数   | 实例                          |
| ---- | ---------------- | ---- | --------------------------- |
| 0    | [ ]              | 0-D  | 一个 0维张量. 一个纯量.              |
| 1    | [D0]             | 1-D  | 一个1维张量的形式[5].               |
| 2    | [D0, D1]         | 2-D  | 一个2维张量的形式[3, 4].            |
| 3    | [D0, D1, D2]     | 3-D  | 一个3维张量的形式 [1, 4, 3].        |
| n    | [D0, D1, … Dn-1] | n-D  | 一个n维张量的形式 [D0, D1, … Dn-1]. |

张量的阶可以使用`tf.rank()`获取到：

~~~python
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
tf.rank(a)  # <tf.Tensor 'Rank:0' shape=() dtype=int32> => 2
~~~

张量的形状可以通过Python中的整数、列表或元祖（int list或tuples）来表示，也或者用`TensorShape`类来表示。如下：

~~~python
# 指定shape是[2, 3]的常量,这里使用了list指定了shape，也可以使用ndarray和TensorShape来指定shape
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], shape=[2, 3])

# 获取shape 方法一：利用tensor的shape属性
a.shape  # TensorShape([Dimension(2), Dimension(3)])

# 获取shape 方法二：利用Tensor的方法get_shape()
a.get_shape()  # TensorShape([Dimension(2), Dimension(3)])

# 获取shape 方法三：利用tf.shape()
tf.shape(a) # <tf.Tensor 'Shape:0' shape=(2, 3) dtype=int32>
~~~

`TensorShape`有一个方法`as_list()`，可以将`TensorShape`转化为python的list。

~~~python
a.get_shape().as_list() # [2, 3]
~~~

同样的我们也可以使用list构建一个TensorShape的对象：

~~~python
ts = tf.TensorShape([2, 3])
~~~

#### Tensor的数据类型

Tensor有一个数据类型属性。可以为一个张量指定下列数据类型中的任意一个类型：

| 数据类型            | Python 类型       | 描述                                     |
| --------------- | --------------- | -------------------------------------- |
| `DT_FLOAT`      | `tf.float32`    | 32 位浮点数.                               |
| `DT_DOUBLE`     | `tf.float64`    | 64 位浮点数.                               |
| `DT_INT64`      | `tf.int64`      | 64 位有符号整型.                             |
| `DT_INT32`      | `tf.int32`      | 32 位有符号整型.                             |
| `DT_INT16`      | `tf.int16`      | 16 位有符号整型.                             |
| `DT_INT8`       | `tf.int8`       | 8 位有符号整型.(此处符号位不算在数值位当中)               |
| `DT_UINT8`      | `tf.uint8`      | 8 位无符号整型.                              |
| `DT_STRING`     | `tf.string`     | 可变长度的字节数组.每一个张量元素都是一个字节数组.             |
| `DT_BOOL`       | `tf.bool`       | 布尔型.(不能使用number类型表示bool类型，但可转换为bool类型) |
| `DT_COMPLEX64`  | `tf.complex64`  | 由两个32位浮点数组成的复数:实部和虚部。                  |
| `DT_COMPLEX128` | `tf.complex128` | 由两个64位浮点数组成的复数:实部和虚部。                  |
| `DT_QINT32`     | `tf.qint32`     | 用于量化Ops的32位有符号整型.                      |
| `DT_QINT8`      | `tf.qint8`      | 用于量化Ops的8位有符号整型.                       |
| `DT_QUINT8`     | `tf.quint8`     | 用于量化Ops的8位无符号整型.                       |

Tensor的数据类型类似于Numpy中的数据类型，但其加入了对string的支持。



##### 设置与获取Tensor的数据类型

设置Tensor的数据类型： 

~~~python
# 方法一
# Tensorflow会推断出类型为tf.float32
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# 方法二
# 手动设置
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32)

# 方法三 (不推荐)
# 设置numpy类型 未来可能会不兼容 
# tf.int32 == np.int32  -> True
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
~~~

获取Tensor的数据类型，可以使用如下方法：

~~~python
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='a')
a.dtype  # tf.float32
print(a.dtype)  # >> <dtype: 'float32'>

b = tf.constant(2+3j)  # tf.complex128 等价于 tf.complex(2., 3.)
print(b.dtype)  # >> <dtype: 'complex128'>

c = tf.constant([True, False], tf.bool)
print(c.dtype)  # <dtype: 'bool'>
~~~

这里需要注意的是一个张量仅允许一种dtype存在，也就是一个张量中每一个数据的数据类型必须一致。

##### 数据类型转化

如果我们需要将一种数据类型转化为另一种数据类型，需要使用`tf.cast()`进行：

~~~python
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], name='a')
# tf.cast(x, dtype, name=None) 通常用来在两种数值类型之间互转
b = tf.cast(a, tf.int16)
print(b.dtype)  # >> <dtype: 'int16'>
~~~

有些类型利用`tf.cast()`是无法互转的，比如string无法转化成为number类型，这时候可以使用以下方法：

~~~python
# 将string转化为number类型 注意：数字字符可以转化为数字
# tf.string_to_number(string_tensor, out_type = None, name = None)
a = tf.constant([['1.0', '2.0', '3.0'], ['4.0', '5.0', '6.0']], name='a')
num = tf.string_to_number(a)
~~~

实数数值类型可以使用cast方法转化为bool类型。

## 2. 节点

图中的节点也可以成为**算子**，它代表一个操作(operation, OP)，一般用来表示数学运算，也可以表示数据输入（feed in）的起点以及输出（push out）的终点，或者是读取/写入持久变量（persistent variable）的终点。常见的节点主要包括以下几种类型：变量、张量元素运算、张量塑形、张量运算、检查点操作、队列和同步操作、张量控制等。

当OP表示数学运算时，每一个运算都会创建一个`tf.Operation`对象。常见的操作，例如生成一个变量或者常量、数值计算均创建`tf.Operation`对象



### 2.1 变量

变量用于存储张量，可以使用list、Tensor等来进行初始化，例如：

~~~python
# 使用纯量0进行初始化一个变量
var = tf.Variable(0)
~~~



### 2.2 张量元素运算

张量元素运算包含几十种常见的运算，比如张量对应元素的相加、相乘等，这里我们介绍以下几种运算：

* **`tf.add()` shape相同的两个张量对应元素相加。等价于`A + B`。**

  ~~~python
  tf.add(1, 2)  # 3
  tf.add([1, 2], [3, 4])  # [4, 6]
  tf.constant([1, 2]) + tf.constant([3, 4])  # [4, 6]
  ~~~

* **`tf.subtract()` shape相同的两个张量对应元素相减。等价于`A - B`。**

  ~~~python
  tf.subtract(1, 2)  # -1
  tf.subtract([1, 2], [3, 4])  # [-2, -2]
  tf.constant([1, 2]) - tf.constant([3, 4])  # [-2, -2]
  ~~~

* **`tf.multiply()` shape相同的两个张量对应元素相乘。等价于`A * B`。**

  ~~~python
  tf.multiply(1, 2)  # 2
  tf.multiply([1, 2], [3, 4])  # [3, 8]
  tf.constant([1, 2]) * tf.constant([3, 4])  # [3, 8]
  ~~~

* **`tf.scalar_mul() `一个纯量分别与张量中每一个元素相乘。等价于 `a * B`**

  ~~~python
  sess.run(tf.scalar_mul(10., tf.constant([1., 2.])))  # [10., 20.]
  ~~~

* **`tf.divide()` shape相同的两个张量对应元素相除。等价于`A / B`。这个除法操作是Tensorflow推荐使用的方法。此方法不接受Python自身的数据结构，例如常量或list等。**

  ~~~python
  tf.divide(1, 2)  # 0.5
  tf.divide(tf.constant([1, 2]), tf.constant([3, 4]))  # [0.33333333, 0.5]
  tf.constant([1, 2]) / tf.constant([3, 4])  # [0.33333333, 0.5]
  ~~~

* **`tf.div()` shape相同的两个张量对应元素相除，得到的结果。不推荐使用此方法。tf.divide与tf.div相比，tf.divide符合Python的语义。例如：**

  ~~~python
  1/2  # 0.5
  tf.divide(tf.constant(1), tf.constant(2))  # 0.5
  tf.div(1/2)  # 0
  ~~~

* **`tf.floordiv()` shape相同的两个张量对应元素相除取整数部分。等价于`A // B`。**

  ~~~python
  tf.floordiv(1, 2)  # 0
  tf.floordiv([4, 3], [2, 5])  # [2, 0]
  tf.constant([4, 3]) // tf.constant([2, 5])  # [2, 0]
  ~~~

* **`tf.mod()` shape相同的两个张量对应元素进行模运算。等价于`A % B`。**

  ~~~python
  tf.mod([4, 3], [2, 5])  # [0, 3]
  tf.constant([4, 3]) % tf.constant([2, 5])  # [0, 3]
  ~~~

上述运算也支持满足一定条件的shape不同的两个张量进行运算。在此不做过多演示。

除此以外还有很多的逐元素操作的函数，例如求平方`tf.square()`、开平方`tf.sqrt`、指数运算、三角函数运算、对数运算等等。



### 2.3 张量运算与塑形

`tf.matmul()` 通常用来做矩阵乘法，张量的阶rank>2，均可使用此方法。

`tf.transpose()` 转置张量。

~~~python
a = tf.constant([[1., 2., 3.], [4., 5., 6.0]])
# tf.matmul(a, b, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False, a_is_sparse=False, b_is_sparse=False, name=None)
# tf.transpose(a, perm=None, name='transpose')
tf.matmul(a, tf.transpose(a))  # 等价于 tf.matmul(a, a, transpose_b=True)
~~~

张量的拼接、切割、形变也是常用的操作：

~~~python
# 沿着某个维度对二个或多个张量进行连接
# tf.concat(values, axis, name='concat')
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 0) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]


# 对输入的张量进行切片
# tf.slice(input_, begin, size, name=None)
'input' is [[[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6]]]
tf.slice(input, [1, 0, 0], [1, 1, 3]) ==> [[[3, 3, 3]]]
tf.slice(input, [1, 0, 0], [1, 2, 3]) ==> [[[3, 3, 3],
                                            [4, 4, 4]]]
tf.slice(input, [1, 0, 0], [2, 1, 3]) ==> [[[3, 3, 3]],
                                           [[5, 5, 5]]]


# 将张量分裂成子张量
# tf.split(value, num_or_size_splits, axis=0, num=None, name='split')
# 'value' is a tensor with shape [5, 30]
# Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1
split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
tf.shape(split0) ==> [5, 4]
tf.shape(split1) ==> [5, 15]
tf.shape(split2) ==> [5, 11]


# 将张量变为指定shape的新张量
# tf.reshape(tensor, shape, name=None)
# tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
# tensor 't' has shape [9]
new_t = tf.reshape(t, [3, 3]) 
# new_t	==> [[1, 2, 3],
#            [4, 5, 6],
#            [7, 8, 9]]
new_t = tf.reshape(new_t, [-1]) # 这里需要注意shape是一阶张量，此处不能直接使用 -1
# tensor 'new_t' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
~~~



### 2.4 其它

其它操作，我们会在之后详述。


----


# 四.常量、变量、占位符

Tensorflow本身并非是一门编程语言，而是一种符号式编程库，用来给C++环境中执行算法的主体提供计算流程的描述。这使得Tensorflow拥有了一些编程语言的特征例如拥有变量、常量，却不是编程语言。Tensorflow用图来完成运算流程的描述。一个图是由OP与Tensor构成，即通过OP生成、消费或改变Tensor。

## 1. 常量

常量是一块只读的内存区域，常量在初始化时就必须赋值，并且之后值将不能被改变。Python并无内置常量关键字，需要我们去实现，而Tensorflow内置了常量方法 `tf.constant()`。

### 1.1 普通常量

普通常量使用`tf.constant()`初始化得到，其有5个参数。

~~~python
constant(value, dtype=None, shape=None, name="Const", verify_shape=False):
~~~

* value是必填参数，即常量的初识值。这里需要注意，这个value可以是Python中的list、tuple以及Numpy中的ndarray，但**不可以是Tensor对象**，因为这样没有意义。
* dtype 可选参数，表示数据类型，value中的数据类型应与与dtype中的类型一致，如果不填写则会根据value中值的类型进行推断。
* shape 可选参数，表示value的形状。如果参数`verify_shape=False`，shape在与value形状不一致时会修改value的形状。如果参数`verify_shape=True`，则要求shape必须与value的shape一致。当shape不填写时，默认为value的shape。

**注意**：`tf.constant()`生成的是一个张量。其类型是tf.Tensor。常量本质上就是一个指向固定张量的符号。



### 1.2 常量存储位置

常量存储在图的定义当中，可以将图序列化后进行查看：

~~~python
const_a = tf.constant([1, 2])
with tf.Session() as sess:
    # tf.Graph.as_graph_def 返回一个代表当前图的序列化的`GraphDef`
    print(sess.graph.as_graph_def()) # 你将能够看到const_a的值
~~~

当常量包含的数据量较大时，会影响图的加载速度。通常较大的数据使用变量或者在之后读取。



### 1.3 序列常量

除了使用`tf.constant()`生成任意常量以外，我们还可以使用一些方法快捷的生成**序列常量**：

~~~python
# 在指定区间内生成均匀间隔的数字
tf.linspace(start, stop, num, name=None) # slightly different from np.linspace
tf.linspace(10.0, 13.0, 4) ==> [10.0 11.0 12.0 13.0] 

# 在指定区间内生成均匀间隔的数字 类似于python中的range
tf.range(start, limit=None, delta=1, dtype=None, name='range')
# 'start' is 3, 'limit' is 18, 'delta' is 3
tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
# 'limit' is 5
tf.range(limit) ==> [0, 1, 2, 3, 4]
~~~



### 1.4 随机数常量

类似于Python中的random模块，Tensorflow也拥有自己的随机数生成方法。可以生成**随机数常量**：

```python
# 生成服从正态分布的随机数
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

# 生成服从截断的正态分布的随机数
# 只保留了两个标准差以内的值，超出的值会被丢掉重新生成
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None,
name=None)

# 生成服从均匀分布的随机值
tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None,
name=None)

# 将输入张量在第一个维度上进行随机打乱
tf.random_shuffle(value, seed=None, name=None)

# 随机的将张量收缩到给定的尺寸
# 注意：不是打乱，是随机的在某个位置开始裁剪指定大小的样本
# 可以利用样本生成子样本
tf.random_crop(value, size, seed=None, name=None)
```

随机数的生成需要随机数种子seed。Tensorflow默认是加了随机数种子的，如果我们希望生成的随机数是固定值，那么可以指定`seed`参数为固定值。

Tensorflow随机数的生成事实上有两个种子在起作用，一个是图级别的，一个是会话级别的。使用`tf.set_random_seed()`可以设置图级的随机种子为固定值，这样在两个不同的会话中执行相同的图时，两个会话中得到的随机数一样。

~~~python
t = tf.random_normal(1)
tf.set_random_seed(123)
with tf.Session() as sess:
  	t1 = sess.run(t)
with tf.Session() as sess:
  	t2 = sess.run(t)
print(t1 == t2)  # True
~~~



### 1.5 特殊常量

~~~python
# 生成指定shape的全0张量
tf.zeros(shape, dtype=tf.float32, name=None)
# 生成与输入的tensor相同shape的全0张量
tf.zeros_like(tensor, dtype=None, name=None,optimize=True)
# 生成指定shape的全1张量
tf.ones(shape, dtype=tf.float32, name=None)
# 生成与输入的tensor相同shap的全1张量
tf.ones_like(tensor, dtype=None, name=None, optimize=True)
# 生成一个使用value填充的shape是dims的张量
tf.fill(dims, value, name=None)
~~~



## 2. 变量

变量用于存取张量，在Tensorflow中主要使用类`tf.Variable()`来实例化一个变量对象，作用类似于Python中的变量。

~~~python
tf.Variable(initial_value=None, trainable=True, collections=None, validate_shape=True, caching_device=None, name=None, variable_def=None, dtype=None, expected_shape=None, import_scope=None)
~~~

initial_value是必填参数，即变量的初始值。可以使用Python中的list、tuple、Numpy中的ndarray、Tensor对象或者其他变量进行初始化。

~~~python
# 使用list初始化
var1 = tf.Variable([1, 2, 3])

# 使用ndarray初始化
var2 = tf.Variable(np.array([1, 2, 3]))

# 使用Tensor初始化
var3 = tf.Variable(tf.constant([1, 2, 3]))

# 使用服从正态分布的随机数Tensor初始化
var4 = tf.Variable(tf.random_normal([3, ]))

# 使用变量var1初始化
var5 = tf.Variable(var1)
~~~

这里需要注意的是：使用`tf.Variable()`得到的对象不是Tensor对象，而是承载了Tensor对象的Variable对象。Tensor对象是一个“流动对象”，可以存在于各种操作中，包括存在与Variable中。所以这也涉及到了如何给变量对象赋值、取值问题。

#### 使用`tf.get_variable()`创建变量

除了使用`tf.Variable()`类实例化一个变量对象以外，还有一种常用的方法来获得一个变量对象：`tf.get_variable()`这是一个方法，不是类，需要注意的是，默认情况下，使用`tf.get_variable()`方法获得一个变量时，其**name不能与已有的name重名**。

~~~python
# 生成一个shape为[3, ]的变量，变量的初值是随机的。
tf.get_variable(name='get_var', shape=[3, ])
# <tf.Variable 'get_var:0' shape=(3,) dtype=float32_ref>
~~~

关于`tf.get_variable()`的更多用法我们会在之后的内容中详解，这里只需要知道这是一种创建变量的方法即可。



### 2.1变量初始化

变量生成的是一个变量对象，只有初始化之后才能参与计算。我们可以使用变量参与图的构建，但在会话中执行时，必须首先初始化。初始化的方法主要有三种。

* 使用变量的属性`initializer`进行初始化：

  例如：

  ~~~python
  var = tf.Variable(tf.constant([1, 2, 3], dtype=tf.float32))
  ...

  with tf.Session() as sess:
      sess.run(var.initializer)
      ...
  ~~~

  ​

* 使用`tf.variables_initializer()`初始化一批变量。

  例如：

  ~~~python
  var1 = tf.Variable(tf.constant([1, 2, 3], dtype=tf.float32))
  var2 = tf.Variable(tf.constant([1, 2, 3], dtype=tf.float32))
  ...

  with tf.Session() as sess:
      sess.run(tf.variables_initializer([var1, var2]))
      ...
  ~~~

  ​

* 使用`tf.global_variables_initialize()`初始化所有变量。

  例如：

  ~~~python
  var1 = tf.Variable(tf.constant([1, 2, 3], dtype=tf.float32))
  var2 = tf.Variable(tf.constant([1, 2, 3], dtype=tf.float32))
  ...

  with tf.Session() as sess:
      sess.run(tf.global_variables_initialize())
      ...
  ~~~

通常，为了简便，第三种方法是首选方法。

在不初始化变量的情况下，也可以使用`tf.Variable.initialized_value()`方法获得其中存储的张量，但我们在运行图时，依然需要初始化变量，否则使用到变量的地方依然会出错。

直接获取变量中的张量：

~~~python
var1 = tf.Variable([1, 2, 3])
tensor1 = var1.initialized_value()
~~~



### 2.2 变量赋值

变量赋值包含两种情况，第一种情况是初始化时进行赋值，第二种是修改变量的值，这时候需要利用赋值函数：

~~~python
A = tf.Variable(tf.constant([1, 2, 3]), dtype=tf.float32)  
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(A))  # >> [1, 2, 3]
    
    # 赋值方法一
    sess.run(tf.assign(A, [2, 3, 4]))  
    print(sess.run(A))  # >> [2, 3, 4]
    
    # 赋值方法二
    sess.run(A.assign([2, 3, 4]))
    print(sess.run(A))  # >> [2, 3, 4]
~~~

**注意：使用`tf.Variable.assign()`或`tf.assign()`进行赋值时，必须要求所赋的值的shape与Variable对象中张量的shape一样、dtype一样。**

除了使用`tf.assign()`以外还可以使用`tf.assign_add()`、`tf.assign_sub()`。

~~~python
A = tf.Variable(tf.constant([1, 2, 3]))
# 将ref指代的Tensor加上value
# tf.assign_add(ref, value, use_locking=None, name=None)
# 等价于 ref.assign_add(value)
A.assign_add(A, [1, 1, 3])  # >> [2, 3, 6]

# 将ref指代的Tensor减去value
# tf.assign_sub(ref, value, use_locking=None, name=None)
# 等价于 ref.assign_sub(value)
A.assign_sub(A, [1, 1, 3])  # >> [0, 1, 0]
~~~



### 2.3 变量操作注意事项

* **注意事项一：**

当我们在会话中运行并输出一个初始化并再次复制的变量时，输出是多少？如下：

```python
W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    sess.run(W) 
```

上面的代码将输出`10`而不是`100`，原因是`w`并不依赖于`W.assign(100)`，`W.assign(100)`产生了一个OP，然而在`sess.run()`的时候并没有执行这个OP，所以并没有赋值。需要改为：

```python
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run([W.initializer, assign_op])
    sess.run(W) 
```



* **注意事项二：**

重复运行变量赋值语句会发生什么？

~~~python
var = tf.Variable(1)
assign_op = var.assign(2 * var)

with tf.Session() as sess:
    sess.run(var.initializer)
    sess.run(assign_op)
    sess.run(var)  # > 2
    
    sess.run(assign_op)  
    sess.run(var)  # > ???
~~~

这里第二次会输出`4`，因为运行了两次赋值op。第一次运行完成时，`var`被赋值为`2`，第二次运行时被赋值为`4`。

那么改为如下情况呢？

~~~python
var = tf.Variable(1)
assign_op = var.assign(2 * var)

with tf.Session() as sess:
    sess.run(var.initializer)
    sess.run([assign_op, assign_op])  
    sess.run(var)  # ???
~~~

这里，会输出`2`。会话run一次，图执行一次，而`sess.run([my_var_times_two, my_var_times_two])`仅仅相当于查看了两次执行结果，并不是执行了两次。

那么改为如下情况呢？

~~~python
var = tf.Variable(1)
assign_op_1 = var.assign(2 * var)
assign_op_2 = var.assign(3 * var)

with tf.Session() as sess:
    sess.run(var.initializer)
    sess.run([assign_op_1, assign_op_2])
    sess.run(var)  # >> ??
~~~

这里两次赋值的Op相当于一个图中的两个子图，其执行顺序不分先后，由于两个子图的执行结果会对公共的变量产生影响，当子图A的执行速度快于子图B时，可能是一种结果，反之是另一种结果，所以这样的写法是不安全的写法，执行的结果是可变。但可以通过控制依赖来强制控制两个子图的执行顺序。

* **注意事项三：**

在多个图中给一个变量赋值：

~~~python
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10)))  # >> 2o
print(sess2.run(W.assign_sub(2)))  # ???

sess1.close()
sess2.close()
~~~

第二个会打印出`8`。因为在两个图中的OP是互不相干的。**每个会话都保留自己的变量副本**，它们分别执行得到结果。



* **注意事项四：**

使用一边量初始化另一个变量时：

~~~python
a = tf.Variable(1)
b = tf.Variable(a)

with tf.Session() as sess:
    sess.run(b.initializer) # 报错
~~~

出错的原因是`a`没有初始化，`b`就无法初始化。所以使用一个变量初始化另一个变量时，可能会是不安全的。为了确保初始化时不会出错，可以使用如下方法：

~~~
a = tf.Variable(1)
b = tf.Variable(a.initialized_value())

with tf.Session() as sess:
    sess.run(b.initializer)
~~~



## 3. 张量、list、ndarray相互转化

Tensorflow本身的执行环境是C++，Python在其中的角色是设计业务逻辑。我们希望尽可能的将所有的操作都在C++环境中进行，以提高执行效率，然而不可避免的是总有一些地方，需要用到Python和Tensorflow进行数据交互。例如`a = tf.Variable([1.0, 2.0])`这句代码事实上使用了Python的list制造了一个原始数据为`[1.0, 2.0]`的变量，然后在会话中执行时会将其传递给Tensorflow的C++环境。这就涉及到了Tensorflow中的数据对象Tensor与Python中的数据对象或相关库的数据对象的交互。即如何将Tensor转化为Python数据类型。

Tensorflow的C++底层使用并扩展了numpy的数据结构，例如我们可以使用`np.float32`代替`tf.float32`等。所以很多时候也会涉及与Numpy中的ndarray的互转。

```python
l = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
print(l)  # >> [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
print(type(l))  # >> list

# 等价的初始化方法一：使用numpy的ndarray初始化
# a = tf.Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
# 等价的初始化方法二：使用张量初始化
# a = tf.Variable(tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
# 等价的初始化方法三：使用list初始化
a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(a)  # >> Tensor("Const_1:0", shape=(2, 3), dtype=int64)
print(type(a))  # <class 'tensorflow.python.framework.ops.Tensor'>

with tf.Session() as sess:
    res = sess.run(a)
    print(res)  # >> [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    print(type(res))  # >> <class 'numpy.ndarray'>
```



有些数据结构不符合张量的特征，也不存在shape等属性，需要加以区分，例如：

```python
t = [1, [2, 3]]
```

`t`表示一个list，但无法转化为张量。



## 4. 占位符

我们定义一个图，就类似于定义一个数学函数。当我们定义一个函数$f(x, y)=x*2+y$时不需要知道其中的自变量$x,y$的值，$x,y$充当了**占位符（placehoders）**。在执行这个函数的时候，再把占位符替换为具体数值。占位符起到了不依赖数据而可以构建函数的目的。使用Tensorflow定义图时，我们也希望图仅仅是用来描述算法的运行过程，而不需要依赖于数据。这样，我们定义的图就更加独立。Tensorflow中，使用`tf.placeholder`构建占位符。

在Tensorflow中占位符也是一个节点op，使用如下方法构建占位符：

~~~python
tf.placeholder(dtype, shape=None, name=None)
~~~

就像使用张量一样，占位符可以直接参与运算：

~~~python
a = tf.placeholder(tf.float32)
b = tf.constant([1, 2, 3])

c = a + b  # 等价于 tf.add(a, b)
~~~

这里我们没有指定`shape`，那么`shape`就是`None`。这意味着可以使用任意shape的张量输入，但这样做不利于调试，假如我们使用一个`shape=[2, ]`的张量输入，那么执行图时一定会报错。所以推荐写上`shape`。

`tf.placehoder`的`shape`可以指定部分维度，例如指定`shape=[None, 10]`，则需要使用第一个维度为任意长度，第二个维度的长度是10的张量。

### 4.1 feed

就如同函数的执行，需要传入自变量一样，图构建好之后，运行图时，需要把占位符用张量来替代(feed)，否则这个图无法运行，如下：

~~~python
a = tf.placeholder(tf.int32)
b = tf.constant([1, 2, 3])

c = a + b  # 等价于 tf.add(a, b)

with tf.Session() as sess:
    sess.run(c, feed_dict={a: [2, 3, 4]})  # >> [3 5 7]
~~~

可以看到建立一个占位符，仅仅只需要输入占位数据的类型即可，即不需要填入具体数据，也不要求写出数据的shape。也就是说可以代替shape可变的数据。

**注意**：`tf.Variable()`并不能代替`tf.placeholder()`。两者的功能完全不同，`tf.Variable()`要求必须有shape，也就是规定了shape必须是固定的。

### 4.2 feed的更多用法

除了`tf.placeholder`可以并且必须使用张量替代以外，很多张量均可以使用`feed_dict`替代，例如：

~~~python
a = multiply(1, 2)

with tf.Session() as sess:
    sess.run(a, feed_dict={a, 10})  # >> 10
~~~

为了保证某个张量能够被其它张量替代，可以使用`tf.Graph.is_feedable(tensor)`检查`tensor`是否可以替代：

~~~python
a = multiply(1, 2)
tf.get_default_graph().is_feedable(a)  # True
~~~

---

# 五.名字与作用域

Tensorflow中所有的Tensor、Operation均拥有自己的名字`name`，是其唯一标识符。在Python中，一个变量可以绑定一个对象，同样的也可以绑定一个Tensor或Operation，但这个变量并不是标识符。Tensorflow使用`name`的好处是可以使我们定义的图在不同的编程环境中均可以使用，例如再C++，Java，Go等API中也可以通过`name`获得Tensor、Variable、Operation，所以`name`可以看做是它们的唯一标识符。Tensorflow使用`name`也可以更方便的可视化。

实际中，使用Python API（或别的API）时，我们通常使用程序语言中的变量来代表Tensor或Operation，而没有指定它们的`name`。这是因为Tensorflow自己命名了`name`。

Tensorflow也有作用域(scope)，用来管理Tensor、Operation的name。Tensorflow的作用域分为两种，一种是variable_scope，另一种是name_scope。简言之，variable_scope主要给variable_name加前缀，也可以给op_name加前缀；name_scope是给op_name加前缀。

## 1. name

Tensor与Operation均有`name`属性，但我们只给Operation进行主动命名，Tensor的`name`由Operation根据自己的`name`与输出数量进行命名（所有的Tensor均由Operation产生）。

例如，我们定义一个常量，并赋予其`name`：

```python
a = tf.constat([1, 2, 3], name='const')
```

这里我们给常量Op定义了一个`name`为`const`。`a`是常量Op的返回值（输出），是一个张量Tensor对象，所以`a`也有自己的`name`，为`const:0`。

可以看到：Operation的name是我们进行命名的，其输出的张量在其后增加了冒号与索引，是TF根据Operation的name进行命名的。

### 1.1 Op的name命名规范

首先Tensor对象的name我们并不能直接进行操作，我们只能给Op设置name。其规范是：**由数字、字母、下划线组成，不能以下划线开头**。

**正确**的命名方式如下：

```python
a1 = tf.constant([1, 2, 3], name='const')
a2 = tf.constant([1, 2, 3], name='123')
a3 = tf.constant([1, 2, 3], name='const_')
```

**错误**的命名方式如下：

```python
a1 = tf.constant([1, 2, 3], name='_const')
a2 = tf.constant([1, 2, 3], name='/123')
a3 = tf.constant([1, 2, 3], name='const:0')
```



### 1.2 Op的name构成

对于一个Op，其`name`就是我们设置的`name`，所以也是由数字、字母、下划线构成的。我们可以通过查看Operation的`name`属性来获得其`name`。

例如：

```python
# 返回一个什么都不做的Op
op = tf.no_op(name='hello')

print(op.name)  # hello
```



### 1.3 Tensor的name构成

有些Op会有返回值，其返回值是一个Tensor。Tensor也有`name`，但与Op不同的是，我们无法直接设置Tensor的`name`，Tensor的`name`是由Op的`name`所决定的。Tensor的`name`相当于在Op的`name`之后加上输出索引。Tensor的`name`由以下三部分构成：

1. op的名称，也就是我们指定的op的name；
2. 冒号；
3. op输出内容的索引，默认从0开始。

例如：

```python
a = tf.constant([1, 2, 3], name='const')

print(a.name)  # const:0
```

这里，我们设置了常量Op的`name`为`const`，这个Op会返回一个Tensor，所以返回的Tensor的`name`就是在其后加上冒号与索引。由于只有一个输出，所以这个输出的索引就是0。

对于两个或多个的输出，其索引依次增加：如下：

```python
key, value = tf.ReaderBase.read(..., name='read')

print(key.name)  # read:0
print(value.name)  # read:1
```



### 1.4 Op与Tensor的默认name

当我们不去设置Op的`name`时，Tensorflow也会默认设置一个`name`，这也正是`name`为可选参数的原因。默认`name`往往与Op的类型相同（默认的`name`并无严格的规律）。

例如：

```python
a = tf.add(1, 2)  
# op name为 `Add`
# Tensor name 为 `Add:0`
```

还有一些特殊的Op，我们没法指定其`name`，只能使用默认的`name`，例如：

```python
init = tf.global_variables_initializer()
print(init.name)  # init
```



### 1.5 重复name的处理方式

Tensorflow并不会严格的规定我们必须设置完全不同的`name`，但Tensorflow同时也不允许存在`name`相同的Op或Tensor，所以当出现了两个Op设置相同的`name`时，Tensorflow会自动给`name`加一个后缀。如下：

```python
a1 = tf.add(1, 2, name='my_add')
a2 = tf.add(3, 4, name='my_add')

print(a1.name)  # my_add:0
print(a2.name)  # my_add_1:0
```

后缀由下划线与索引组成（注意区分Tensor的name后缀与冒号索引）。从重复的第二个`name`开始加后缀，后缀的索引从1开始。

当我们不指定`name`时，使用默认的`name`也是相同的处理方式：

```python
a1 = tf.add(1, 2)
a2 = tf.add(3, 4)

print(a1.name)  # Add:0
print(a2.name)  # Add_1:0
```



### 1.6 不同图中相同操作name

当我们构建了两个或多个图的时候，如果这两个图中有相同的操作或者相同的`name`时，并不会互相影响。如下：

```python
g1 = tf.Graph()
with g1.as_default():
    a1 = tf.add(1, 2, name='add')
    print(a1.name)  # add:0

g2 = tf.Graph()
with g2.as_default():
    a2 = tf.add(1, 2, name='add')
    print(a2.name)  # add:0
```

可以看到两个图中的`name`互不影响。并没有关系。



## 2. 通过name获取Op与Tensor

上面，我们介绍了`name`可以看做是Op与Tensor的标识符，其实`name`不仅仅只是标识其唯一性的工具，也可以利用`name`获取到Op与Tensor。（我们可以不借助Python中的变量绑定对象的方式操作Op与Tensor，但是写法很复杂。）

例如，一个计算过程如下：

```python
g1 = tf.Graph()
with g1.as_default():
	a = tf.add(3, 5)
	b = tf.multiply(a, 10)
    
with tf.Session(graph=g1) as sess:
    sess.run(b)  # 80
```

我们也可以使用如下方式，两种方式结果一样：

```python
g1 = tf.Graph()
with g1.as_default():
    tf.add(3, 5, name='add')
    tf.multiply(g1.get_tensor_by_name('add:0'), 10, name='mul')
    
with tf.Session(graph=g1) as sess:
    sess.run(g1.get_tensor_by_name('mul:0'))  # 80
```

这里使用了`tf.Graph.get_tensor_by_name`方法。可以根据`name`获取Tensor。其返回值是一个Tensor对象。这里要注意Tensor的`name`必须写完整。

利用`name`也可以获取到相应的Op，这里需要使用`tf.Graph.get_operation_by_name`方法。上述例子中，我们在会话中运行的是乘法操作的返回值`b`。运行`b`的时候，与其相关的依赖，包括乘法Op也运行了，当我们不需要返回值时，我们在会话中可以直接运行Op，而不是Tensor。

例如：

```python
g1 = tf.Graph()
with g1.as_default():
    tf.add(3, 5, name='add')
    tf.multiply(g1.get_tensor_by_name('add:0'), 10, name='mul')
    
with tf.Session(graph=g1) as sess:
    sess.run(g1.get_operation_by_name('mul'))  # None
```

在会话中，fetch一个Tensor，会返回一个Tensor，fetch一个Op，返回`None`。



## 3. name_scope

name_scope可以用来给op_name、tensor_name加前缀。其目的是区分功能模块，可以更方便的在TensorBoard中可视化，同时也便于管理name，以及便于持久化和重新加载。

name_scope使用`tf.name_scope()`创建，返回一个上下文管理器。name_scope的参数`name`可以是字母、数字、下划线，不能以下划线开头。类似于Op的`name`的命名方式。

`tf.name_scope()`的详情如下：

~~~python
tf.name_scope(
    name,  # 传递给Op name的前缀部分
    default_name=None,  # 默认name
    values=None)  # 检测values中的tensor是否与下文中的Op在一个图中
~~~

注意：`values`参数可以不填。当存在多个图时，可能会出现在当前图中使用了在别的图中的Tensor的错误写法，此时如果不在Session中运行图，并不会报错，而填写到了`values`参数的中的Tensor都会检测其所在图是否为当前图，提高安全性。

使用`tf.name_scope()`的例子：

```python
a = tf.constant(1, name='const')
print(a.name)  # >> const:0

with tf.name_scope('scope_name') as name:
  	print(name)  # >> scope_name/
  	b = tf.constant(1, name='const')
    print(b.name)  # >> scope_name/const:0
```

在一个name_scope的作用域中，可以填写name相同的Op，但Tensorflow会自动加后缀，如下：

~~~python
with tf.name_scope('scope_name') as name:
    a1 = tf.constant(1, name='const')
    print(b.name)  # scope_name/const:0
    a2 = tf.constant(1, name='const')
    print(c.name)  # scope_name/const_1:0
~~~

#### 多个name_scope

我们可以指定任意多个name_scope，并且可以填写相同`name`的两个或多个name_scope，但Tensorflow会自动给name_scope的name加上后缀：

如下：

```python
with tf.name_scope('my_name') as name1:
  	print(name1)  # >> my_name/
    
with tf.name_scope('my_name') as name2:
  	print(name2)  #>> my_name_1/
```



### 3.1 多级name_scope

name_scope可以嵌套，嵌套之后的name包含上级name_scope的name。通过嵌套，可以实现多样的命名，如下：

```python
with tf.name_scope('name1'):
  	with tf.name_scope('name2') as name2:
      	print(name2)  # >> name1/name2/
```

不同级的name_scope可以填入相同的name（本质上不同级的name_scope不存在同名），如下：

~~~python
with tf.name_scope('name1') as name1:
    print(name1)  # >> name1/
  	with tf.name_scope('name1') as name2:
      	print(name2)  # >> name1/name1/
~~~

在多级name_scope中，op的name会被加上一个前缀，这个前缀取决于所在的name_scope。不同级中的name因为其前缀不同，所以不可能重名，如下：

```python
with tf.name_scope('name1'):
  	a = tf.constant(1, name='const')
    print(a.name)  # >> name1/const:0
  	with tf.name_scope('name2'):
      	b = tf.constant(1, name='const')
    	print(b.name)  # >> name1/name2/const:0
```



### 3.2 name_scope的作用范围

使用name_scope可以给op_name加前缀，但不包括`tf.get_variable()`创建的变量Op，如下所示：

```python
with tf.name_scope('name'):
  	var = tf.Variable([1, 2], name='var')
    print(var.name)  # >> name/var:0
    var2 = tf.get_variable(name='var2', shape=[2, ])
    print(var2.name)  # >> var2:0
```



### 3.3 注意事项

1. 从外部传入的Tensor，并不会在name_scope中加上前缀。例如：

   ~~~python
   a = tf.constant(1, name='const')
   with tf.name_scope('my_name', values=[a]):
       print(a.name)  # >> const:0
   ~~~

   ​

2. Op与name_scope的`name`中可以使用`/`，但`/`并不是`name`的构成，还是区分命名空间的符号，不推荐直接使用`/`。

   ​

3. name_scope的`default_name`参数可以在函数中使用。name_scope返回的str类型的scope可以作为`name`传给Op的`name`，这样做的好处是返回的Tensor的name反映了其所在的模块。例如：

   ~~~python
   def my_op(a, b, c, name=None):
       with tf.name_scope(name, "MyOp", [a, b, c]) as scope:
           a = tf.convert_to_tensor(a, name="a")
           b = tf.convert_to_tensor(b, name="b")
           c = tf.convert_to_tensor(c, name="c")
           # Define some computation that uses `a`, `b`, and `c`.
           return foo_op(..., name=scope)
   ~~~

   ​

## 4. variable_scope

variable_scope也可以用来给name加前缀，包括variable_name与op_name都可以。与name_scope相比，variable_scope功能要更丰富，最重要的是其可以给get_variable()创建的变量加前缀。

variable_scope使用`tf.variable_scope()`创建，返回一个上下文管理器。name_scope的参数`name`可以是字母、数字、下划线，不能以下划线开头。类似于变量的参数`name`以及name_scope的命名。

`tf.variable_scope`的详情如下:

~~~python
variable_scope(name_or_scope,  # 可以是name或者别的variable_scope
               default_name=None,
               values=None,
               initializer=None,
               regularizer=None,
               caching_device=None,
               partitioner=None,
               custom_getter=None,
               reuse=None,
               dtype=None,
               use_resource=None):
~~~



### 4.1 给op_name加前缀

variable_scope包含了name_scope的功能，默认的variable_scope的`name`等于其中的name_scope的`name`。如下：

```python
g = tf.Graph()
with g.as_default():
    with tf.variable_scope('abc') as scope:
      	# 输出variable_scope的`name`
        print(scope.name)  # >> abc
        
        n_scope = g.get_name_scope()
        # 输出name_scope的`name`
        print(n_scope)  # >> abc
```

在variable_scope下也可以给Op与Tensor加前缀：

~~~python
with tf.variable_scope('abc') as scope:
    a = tf.constant(1, name='const')
    print(a.name)  # >> abc/const:0
~~~



### 4.2 给variable_name加前缀

variable_scope与name_scope最大的不同就是，variable_scope可以给使用`tf.get_variable()`创建的变量加前缀，如下：

```python
with tf.variable_scope('my_scope'):
  	var = tf.get_variable('var', shape=[2, ])
    print(var.name)  # >> my_scope/var:0
```

`tf.get_variable()`创建变量时，必须有`name`与`shape`。`dtype`可以不填，默认是`tf.float32`。同时，使用`tf.get_variable()`创建变量时，**name不能填入重复的**。

以下写法是错误的：

~~~python
a = tf.get_variable('abcd', shape=[1])
b = tf.get_variable('abcd', shape=[1])  # ValueError
~~~



### 4.3 同名variable_scope

创建两个或多个variable_scope时可以填入相同的`name`，此时相当于创建了一个variable_scope与两个name_scope。

~~~python
g = tf.Graph()
with g.as_default():
    with tf.variable_scope('abc') as scope:
        print(scope.name)  # >> abc
        n_scope = g.get_name_scope()
        print(n_scope)  # >> abc
        
    with tf.variable_scope('abc') as scope:
        print(scope.name)  # >> abc
        n_scope = g.get_name_scope()
        print(n_scope)  # >> abc_1
~~~

同名的variable_scope，本质上属于一个variable_scope，不允许通过`tf.get_variable`创建相同name的Variable。下面的代码会抛出一个ValueError的错误：

```python
with tf.variable_scope('s1'):
  	vtf.get_variable('var')

with tf.variable_scope('s1'):
  	# 抛出错误
  	tf.get_variable('var')
```

使用一个variable_scope初始化另一个variable_scope。这两个variable_scope的name一样，name_scope的name不一样。相当于是同一个variable_scope。如下：

```python
with tf.variable_scope('my_scope') as scope1:
  	print(scope1.name)  # >> my_scope
    
with tf.variable_scope(scope1) as scope2:
 	print(scope2.name)  # >> my_scope
```



#### variable_scope的reuse参数

创建variable_scope时，默认的`reuse`参数为`None`，当设置其为`True`时，此处的variable_scope成为了共享变量scope，即可以利用`tf.get_variable()`共享其它同名的但`reuse`参数为`None`的variable_scope中的变量。此时`tf.get_variable()`的作用成为了“获取同名变量”，而不能够创建变量（尝试创建一个新变量会抛出ValueError的错误）。

**注意：`reuse`参数的取值是`None`或者`True`，不推荐使用`False`代替`None`。**

**`tf.get_variable()`配合variable_scope使用，才能够发挥其create与get变量的双重能力。**

例如：

```python
with tf.variable_scope('my_scope') as scope1:
  	# 默认情况下reuse=None
  
    # 创建变量
  	var = tf.get_variable('var', shape=[2, 3])

with tf.variable_scope('my_scope', reuse=True) as scope2:
    
    # 使用tf.get_variable()获取变量
  	var2 = tf.get_variable('var')
    var3 = tf.get_variable('var')
    
print(var is var2)  # >> True
print(var is var3)  # >> True
```

#### 使用scope.reuse_variables分割variable_scope

使用`scope.reuse_variables()`可以将一个variable_scope分割成为可以创建变量与可以重用变量的两个块。例如：

~~~python
with tf.variable_scope('my_scope') as scope:
    a1 = tf.get_variable('my_var', shape=[1, 2])
    scope.reuse_variables()
    a2 =tf.get_variable('my_var')
    
    print(a1 is a2)  # >> True
~~~



### 4.4 多级变量作用域

我们可以在一个作用域中，使用另一个作用域，这时候作用域中的name也会叠加在一起，如下：

```python
with tf.variable_scope('first_scope') as first_scope:
  	print(first_scope.name)  # >> first_scope
  	with tf.variable_scope('second_scope') as second_scope:
      	print(second_scope.name)  # >> first_scope/second_scope
        print(tf.get_variable('var', shape=[1, 2]).name)  
        # >> first_scope/second_scope/var:0
```



#### 跳过作用域

如果在开启的一个变量作用域里使用之前预定义的一个作用域，则会跳过当前变量的作用域，保持预先存在的作用域不变：

```python
with tf.variable_scope('outside_scope') as outside_scope:
  	print(outside_scope.name)  # >> outside_scope

with tf.variable_scope('first_scope') as first_scope:
  	print(first_scope.name)  # >> first_scope
  	print(tf.get_variable('var', shape=[1, 2]).name)   # >> first_scope/var:0
   
  	with tf.variable_scope(outside_scope) as second_scope:
      	print(second_scope.name)  # >> outside_scope
      	print(tf.get_variable('var', shape=[1, 2]).name)  # >> outside_scope/var:0
```

#### 多级变量作用域中的reuse

在多级变量作用域中，规定外层的变量作用域设置了`reuse=True`，内层的所有作用域的`reuse`必须设置为`True`（设置为其它无用）。

多级变量作用域中，使用`tf.get_variable()`的方法如下：

```python
# 定义
with tf.variable_scope('s1') as s1:
    tf.get_variable('var', shape=[1,2])
    with tf.variable_scope('s2') as s2:
        tf.get_variable('var', shape=[1,2])
        with tf.variable_scope('s3') as s3:
            tf.get_variable('var', shape=[1,2])

# 使用
with tf.variable_scope('s1', reuse=True) as s1:
    v1 = tf.get_variable('var')
    with tf.variable_scope('s2', reuse=None) as s2:
        v2 = tf.get_variable('var')
        with tf.variable_scope('s3', reuse=None) as s3:
            v3 = tf.get_variable('var')
```



### 4.5 变量作用域的初始化

variable_scope可以在创建时携带一个初始化器。其作用是将在其中的变量自动使用初始化器的方法进行初始化。方法如下：

```python
# 直接使用tf.get_variable()得到的是随机数
var1 = tf.get_variable('var1', shape=[3, ])
var1.eval()
# 输出的可能值：
# array([-0.92183685, -0.078825  , -0.61367416], dtype=float32)
```

```python
# 使用variable_scope的初始化器
with tf.variable_scope(
  					 'scope', 
  					 initializer=tf.constant_initializer(1)):
  	var2 = tf.get_variable('var2', shape=[3, ])
	var1.eval()  # 输出 [ 1.  1.  1.]
```

常见的初始化器有：

```python
# 常数初始化器
tf.constant_initializer(value=0, dtype=tf.float32)
# 服从正态分布的随机数初始化器
tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
# 服从截断正态分布的随机数初始化器
tf.truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None, dtype=tf.float32)
# 服从均匀分布的随机数初始化器
tf.random_uniform_initializer(minval=0, maxval=None, seed=None, dtype=tf.float32)
# 全0初始化器
tf.zeros_initializer(dtype=tf.float32)
# 全1初始化器
tf.ones_initializer(dtype=tf.float32)
```

---

# 六.文件IO与模型存取

当我们已经训练好一个模型时，需要把模型保存下来，这样训练的模型才能够随时随地的加载与使用。模型的保存和加载分为两部分，一部分是模型定义的存取即**图的存取**，另一部分中图中的**变量的存取**（常量存储在图中）。Tensorflow可以分别完成两部分的内容的存取。

图与变量的存取均涉及到了文件IO操作。Tensorflow拥有独立的文件IO API。当然我们也可以使用Python中的文件IO方式，但Tensorflow的API的功能更加丰富。



## 1. Tensorflow的文件IO模块

在Tensorflow中，文件IO的API类似于Python中对file对象操作的API。但Tensorflow的文件IO会调用C++的接口，实现文件操作。同时Tensorflow的文件IO模块也支持更多的功能，包括操作本地文件、Hadoop分布式文件系统(HDFS)、谷歌云存储等。

Tensorflow的文件IO模块为`tf.gfile`。其中提供了丰富的文件操作的API，例如文件与文件夹的创建、修改、复制、删除、查找、统计等。这里我们简单介绍几种常用的文件操作方法。



### 1.1 打开文件

打开并操作文件首先需要创建一个文件对象。这里有两种方法进行操作：

* `tf.gfile.GFile`。也可以使用`tf.gfile.Open`，两者是等价的。此方法创建一个文件对象，返回一个上下文管理器。

  用法如下：

  ~~~python
  tf.gfile.GFile(name, mode='r')
  ~~~

  输入一个文件名进行操作。参数`mode`是操作文件的类型，有`"r", "w", "a", "r+", "w+", "a+"`这几种。分别代表只读、只写、增量写、读写、读写（包括创建）、可读可增量写。默认情况下读写操作都是操作的文本类型，如果需要写入或读取bytes类型的数据，就需要在类型后再加一个`b`。这里要注意的是，其与Python中文件读取的`mode`类似，但不存在`"t","U"`的类型（不加`b`就等价于Python中的`t`类型）。

* `tf.gfile.FastGFile`。与`tf.gfile.GFile`用法、功能都一样(旧版本中`tf.gfile.FastGFile`支持无阻塞的读取，`tf.gfile.GFile`不支持。目前的版本都支持无阻塞读取)。一般的，使用此方法即可。

例如：

~~~python
# 可读、写、创建文件
with tf.gfile.GFile('test.txt', 'w+') as f:
    ...
    
# 可以给test.txt追加内容
with tf.gfile.Open('test.txt', 'a') as f:
    ...
    
# 只读test.txt
with tf.gfile.FastGFile('test.txt', 'r') as f:
    ...
    
# 操作二进制格式的文件
with tf.gfile.FastGFile('test.txt', 'wb+') as f:
    ...
~~~



### 1.2 文件读取

文件读取使用文件对象的`read`方法。（这里我们以`FastGFile`为例，与`GFile`一样）。文件读取时，会有一个指针指向读取的位置，当调用`read`方法时，就从这个指针指向的位置开始读取，调用之后，指针的位置修改到新的未读取的位置。`read`的用法如下：

~~~python
# 返回str类型的内容
tf.gfile.FastGFile.read(n=-1)
~~~

当参数`n=-1`时，代表读取整个文件。`n!=-1`时，代表读取`n`个bytes长度。

例如：

~~~python
with tf.gfile.FastGFile('test.txt', 'r') as f:
    f.read(3)  # 读取前3个bytes
    f.read()  # 读取剩下的所有内容
~~~

如果我们需要修改文件指针，只读部分内容或跳过部分内容，可以使用`seek`方法。用法如下：

~~~python
tf.gfile.FastGFile.seek(
    offset=None,  # 偏移量 以字节为单位
    whence=0,  # 偏移其实位置 0表示从文件头开始(正向) 1表示从当前位置开始(正向) 2表示从文件末尾开始(反向)
    position=None  # 废弃参数 使用`offset`参数替代
)
~~~

例如：

~~~python
with tf.gfile.FastGFile('test.txt', 'r') as f:
    f.seed(3)  # 跳过前3个bytes
    f.read()  # 读取剩下的所有内容
~~~

**注意：读取文件时，默认的（不加`b`的模式）会对文件进行解码。会将bytes类型转换为UTF-8类型，如果读入的数据编码格式不是UTF-8类型，则在解码时会出错，这时需要使用二进制读取方法。**

除此以外，还可以使用`reafline`方法对文件进行读取。其可以读取以`\n`为换行符的文件的一行内容。例如：

~~~python
with tf.gfile.FastGFile('test.txt', 'r') as f:
    f.readline()  # 读取一行内容(包括行末的换行符)
    f.readlines()  # 读取所有行，返回一个list，list中的每一个元素都是一行
~~~

以行为单位读取内容时，还可以使用`next`方法或是使用生成器来读取。如下：

~~~python
with tf.gfile.FastGFile('test.txt', 'r') as f:
    f.next()  # 读取下一行内容
    
with tf.gfile.FastGFile('test.txt', 'r') as f:
  	# 二进制数据首先会将其中的代表`\n`的字符转换为\n，然后会以\n作为分隔符生成list
    lines = [line for line in f]  
~~~

**注意：如果没有使用with承载上下文管理器，文件读取完毕之后，需要显示的使用`close`方法关闭文件IO。**



### 1.3 其它文件操作

* **文件复制**：`tf.gfile.Copy(oldpath, newpath, overwrite=False)`
* **删除文件**：`tf.gfile.Remove(filename)`
* **递归删除**：`tf.gfile.DeleteRecursively(dirname)`
* **判断路径是否存在**：`tf.gfile.Exists(filename)`  # filename可指代路径
* **判断路径是否为目录**：`tf.gfile.IsDirectory(dirname)`
* **返回当前目录下的内容**：`tf.gfile.ListDirectory(dirname)`  # 不递归 不显示'.'与'..'
* **创建目录**：`tf.gfile.MkDir(dirname)`  # 其父目录必须存在
* **创建目录**：`tf.gfile.MakeDirs(dirname)`  # 任何一级目录不存在都会进行创建
* **文件改名**：`tf.gfile.Rename(oldname, newname, overwrite=False)`
* **统计信息**：`tf.gfile.Stat(filename)`
* **文件夹遍历**：`tf.gfile.Walk(top, in_order=True)`  # 默认广度优先
* **文件查找**：`tf.gfile.Glob(filename)`  # 支持pattern查找




## 2. 图存取

图是算法过程的描述工具，当我们在某个文件中定义了一个图的时候，也就定义了一个算法，当我们需要运行这个算法时，可以直接找到定义此图的文件，就能操作它。所以，通常地我们并不需要保存图，但如果我们希望一个图能够脱离编程语言环境去使用，这时候就需要将其序列化成为某种固定的数据格式，这时候就可以实现跨语言的操作。

图是由一系列Op与Tensor构成的，我们可以通过某种方法对这些Op与Tensor进行描述，在Tensorflow中这就是'图定义'`GraphDef`。图的存取本质上就是`GraphDef`的存取。



### 2.1 图的保存

图的保存方法很简单，只需要将图的定义保存即可。所以：

**第一步，需要获取图定义。**

可以使用`tf.Graph.as_graph_def`方法来获取序列化后的图定义`GraphDef`。

例如：

~~~python
with tf.Graph().as_default() as graph:
    v = tf.constant([1, 2])
    print(graph.as_graph_def())
~~~

输出内容：

~~~python
输入内容：
node {
  name: "Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 2
          }
        }
        tensor_content: "\001\000\000\000\002\000\000\000"
      }
    }
  }
}
versions {
  producer: 24
}
~~~

还可以使用绑定图的会话的`graph_def`属性来获取图的序列化后的定义，例如：

~~~python
with tf.Graph().as_default() as graph:
    v = tf.constant([1, 2])
    print(graph.as_graph_def())
    
with tf.Session(graph=graph) as sess:
    sess.graph_def == graph.as_graph_def()  # True
~~~

**注意：**当会话中加入Op时，`sess.graph_def == graph.as_graph_def()`不再成立。在会话中graph_def会随着Op的改变而改变。

获取了图的定义之后，便可以去保存图。

**第二步：保存图的定义**

保存图的定义有两种方法，第一种直接将图存文件。利用上面我们学习的文件IO操作即可完成。同时，Tensorflow也提供了专门的保存图的方法，更加便捷。

* 方法一：

  直接创建一个文件保存图定义。如下：

  ~~~python
  with tf.Graph().as_default() as g:
      tf.Variable([1, 2], name='var')

      with tf.gfile.FastGFile('test_model.pb', 'wb') as f:
          f.write(g.as_graph_def().SerializeToString())
  ~~~

  `SerializeToString`是将str类型的图定义转化为二进制的proto数据。

* 方法二：

  使用Tensorflow提供的`tf.train.write_graph`进行保存。使用此方法还有一个好处，就是可以直接将图传入即可。用法如下：

  ~~~python
  tf.train.write_graph(
      graph_or_graph_def, # 图或者图定义
      logdir,  # 存储的文件路径
      name,   # 存储的文件名
      as_text=True)  # 是否作为文本存储
  ~~~

  例如，'方法一'中的图也可以这样保存：

  ~~~python
  with tf.Graph().as_default() as g:
      tf.Variable([1, 2], name='var')
      
      tf.train.write_graph(g, '', 'test_model.pb', as_text=False)
  ~~~

  这些参数`as_text`的值为`False`，即保存为二进制的proto数据。此方法等价于'方法一'。

  当`as_text`值为`True`时，保存的是str类型的数据。通常推荐为`False`。



### 2.2 图的获取

图的获取，即将保存的图的节点加载到当前的图中。当我们保存一个图之后，这个图可以再次被获取到，这个操作是很有用的，例如我们在当前编程语言下构建了一个图，而这个图可能会应用到其它编程语言环境下，就需要将图保存并在需要的时候再次获取。

图的获取步骤如下：

1. 读取保存的图的数据
2. 创建`GraphDef`对象
3. 导入`GraphDef`对象到当前图中

具体如下：

~~~python
with tf.Graph().as_default() as new_graph:
    with tf.gfile.FastGFile('test_model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)
~~~

这里`ParseFromString`是protocal message的方法，用于将二进制的proto数据读取成`GraphDef`数据。`tf.import_graph_def`用于将一个图定义导入到当前的默认图中。

这里有一个问题，那就是导入图中的Op与Tensor如何获取到呢？`tf.import_graph_def`都已经帮我们想到这些问题了。这里，我们可以了解下`tf.import_graph_def`的用法：

~~~python
tf.import_graph_def(
    graph_def,  # 将要导入的图的图定义
    input_map=None,  # 替代导入的图中的Tensor
    return_elements=None,  # 返回指定的OP或Tensor(可以使用新的变量绑定)
    name=None,  # 被导入的图中Op与Tensor的name前缀 默认是'import'
    op_dict=None, 
    producer_op_list=None):
~~~

**注意**：当`input_map`不为None时，`name`必须不为空。

**注意**：当`return_elements`返回Op时，在会话中执行返回为`None`。

当然了，我们也可以使用`tf.Graph.get_tensor_by_name`与`tf.Graph.get_operation_by_name`来获取Tensor与Op，但要注意加上name前缀。

如果图在保存时，存为文本类型的proto数据，即`tf.train.write_graph`中的参数`as_text`为`True`时，获取图的操作稍有不同。即解码时不能使用`graph_def.ParseFromString`进行解码，而需要使用`protobuf`中的`text_format`进行操作，如下：

~~~python
from google.protobuf import text_format

with tf.Graph().as_default() as new_graph:
    with tf.gfile.FastGFile('test_model.pb', 'r') as f:
        graph_def = tf.GraphDef()
        # graph_def.ParseFromString(f.read())
        text_format.Merge(f.read(), graph_def)
        tf.import_graph_def(graph_def)
~~~



## 3. 变量存取

变量存储是把模型中定义的变量存储起来，不包含图结构。另一个程序使用时，首先需要重新创建图，然后将存储的变量导入进来，即模型加载。变量存储可以脱离图存储而存在。

变量的存储与读取，在Tensorflow中叫做检查点存取，变量保存的文件是检查点文件(checkpoint file)，扩展名一般为.ckpt。使用`tf.train.Saver()`类来操作检查点。



### 3.1 变量存储

变量是在图中定义的，但实际上是会话中存储了变量，即我们在运行图的时候，变量才会真正存在，且变量在图的运行过程中，值会发生变化，所以我们需要**在会话中保存变量**。保存变量的方法是`tf.train.Saver.save()`。

这里需要注意，通常，我们可以在图定义完成之后初始化`tf.train.Saver()`。`tf.train.Saver()`在图中的位置很重要，在其之后的变量不会被存储。

创建Saver对象之后，此时并不会保存变量，我们还需要指定会话运行到什么时候时再去保存变量，需要使用`tf.train.Saver.save()`进行保存。

例如：

~~~python
with tf.Graph().as_default() as g:
    var1 = tf.Variable(1)
    saver = tf.train.Saver()
    
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(var1.assign_add(1))
    saver.save(sess, './model.cpkt')
~~~

`tf.train.Saver.save()`有两个必填参数，第一个是会话，第二个是存储路径。保存变量之和，会在指定目录下生成四个文件。分别是checkpoint文件、model.ckpt.data-00000-of-00001文件、model.ckpt.index文件、model.ckpt.meta文件。这四个文件的作用分别是：

- checkpoint：保存当前模型的位置与最近的5个模型的信息。这里我们只保存了一个模型，所有只有一个模型信息，也是当前模型的信息。
- model.ckpt.data-00000-of-00001：保存了当前模型变量的数据。
- model.ckpt.meta：保存了`MetaGraphDef`，可以用于恢复saver对象。
- model.ckpt.index：辅助model.ckpt.meta的数据文件。

#### 循环迭代算法时存储变量

实际中，我们训练一个模型，通常需要迭代较多的次数，迭代的过程会用去很多的时间，为了避免出现意外情况（例如断电、死机），我们可以每迭代一定次数，就保存一次模型，如果出现了意外情况，就可以快速恢复模型，不至于重新训练。

如下，我们需要迭代1000次模型，每100次迭代保存一次：

~~~python
with tf.Graph().as_default() as g:
    var1 = tf.Variable(1, name='var')
    saver = tf.train.Saver()
    
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(1, 1001):
        sess.run(var1.assign_add(1))
        if i % 100 == 0:
            saver.save(sess, './model2.cpkt')
            
    saver.save(sess, './model2.cpkt')   
~~~

这时候每次存储都会覆盖上次的存储信息。但我们存储的模型并没有与训练的次数关联起来，我们并不知道当前存储的模型是第几次训练后保存的结果，如果中途出现了意外，我们并不知道当前保存的模型是什么时候保存下的。所以通常的，我们还需要将训练的迭代的步数进行标注。在Tensorflow中只需要给save方法加一个参数即可，如下：

~~~python
with tf.Graph().as_default() as g:
    var1 = tf.Variable(1, name='var')
    saver = tf.train.Saver()
    
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(1, 1001):
        sess.run(var1.assign_add(1))
        if i % 100 == 0:
            saver.save(sess, './model2.cpkt', global_step=i)  # 
            
    saver.save(sess, './model2.cpkt', 1000)
~~~

这里，我们增加了给`saver.save`增加了`global_step`的参数，这个参数可以是一个0阶Tensor或者一个整数。之后，我们生成的保存变量文件的文件名会加上训练次数，并同时保存最近5次的训练结果，即产生了16个文件。包括这五次结果中每次训练结果对应的data、meta、index三个文件，与一个checkpoint文件。



### 3.2 变量读取

变量读取，即加载模型数据，为了保证数据能够正确加载，必须首先将图定义好，而且必须与保存时的图定义一致。这里“一致”的意思是图相同，对于Python句柄等Python环境中的内容可以不同。

下面我们恢复上文中保存的图：

~~~python
with tf.Graph().as_default() as g:
    var1 = tf.Variable(1, name='var')
    saver = tf.train.Saver()

with tf.Session(graph=g) as sess:
    sess2.run(tf.global_variables_initializer())
    saver.restore(sess, './model.ckpt')
    print(sess.run(var))  # >> 1001
~~~

除了使用这种方法恢复变量以外，还可以借助meta数据恢复，

例如：

~~~python
with tf.Graph().as_default() as g:
    var1 = tf.Variable(1, name='var')
    # saver = tf.train.Saver()

with tf.Session(graph=g) as sess:
    sess2.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./log/model.ckpt.meta')
    saver.restore(sess, './model.ckpt')
    print(sess.run(var))  # >> 1001
~~~

不仅如此，使用meta数据进行恢复时，meta数据中也包含了图定义。所以在我们没有图定义的情况下，也可以使用meta数据进行恢复，例如：

~~~python
with tf.Graph().as_default() as g:
    op = tf.no_op()

with tf.Session(graph=g) as sess:
    sess2.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./log/model.ckpt.meta')
    saver.restore(sess, './model.ckpt')
    print(sess.run(g.get_tensor_by_name('var:0')))  # >> 1001
~~~

所以，通过meta数据不仅能够恢复变量，也能够恢复图定义。

在恢复数据时，可能存在多个模型保存的文件，为了简便可以使用`tf.get_checkpoint_state()`来获取最新训练的模型。其返回一个`CheckpointState`对象，可以使用这个对象`model_checkpoint_path`属性来获得最新模型的路径。例如：

~~~python
...
with tf.Session() as sess:
    sess1.run(tf.global_variables_initializer())
    
    ckpt = tf.train.get_checkpoint_state('./log')
    saver.restore(sess1, ckpt.model_checkpoint_path)
~~~



### 3.3 注意事项

**注意事项1**

当一个图对应多个会话时，在不同的会话中使用图的Saver对象保存变量时，并不会互相影响。

例如，对于两个会话`sess1`、`sess2`分别操作一个图`g`中的变量，并存储在不同文件中：

~~~python
with tf.Graph().as_default() as g:
    var = tf.Variable(1, name='var')
    saver = tf.train.Saver()
    
    
with tf.Session(graph=g) as sess1:
    sess1.run(tf.global_variables_initializer())
    sess1.run(var.assign_add(1))
    
    saver.save(sess1, './model1/model.ckpt')
    
    
with tf.Session(graph=g) as sess2:
    sess2.run(tf.global_variables_initializer())
    sess2.run(var.assign_sub(1))
    
    saver.save(sess2, './model2/model.ckpt')
~~~

当我们分别加载变量时，可以发现，并没有互相影响。如下：

~~~python
with tf.Session(graph=g) as sess3:
    sess3.run(tf.global_variables_initializer())
    saver.restore(sess3, './model1/model.ckpt')
    print(sess3.run(var))  # >> 2
    
with tf.Session(graph=g) as sess4:
    sess4.run(tf.global_variables_initializer())
    saver.restore(sess4, './model1/model.ckpt')
    print(sess4.run(var))  # >> 0
~~~

**注意事项2**

当我们在会话中恢复变量时，必须要求会话所绑定的图与所要恢复的变量所代表的图一致，这里我们需要知道什么样的图时一致。

例如：

~~~python
with tf.Graph().as_default() as g1:
    a = tf.Variable([1, 2, 3])
    
with tf.Graph().as_default() as g2:
    b = tf.Variable([1, 2, 3])
~~~

这两个图是一致的，虽然其绑定的Python变量不同。

~~~python
with tf.Graph().as_default() as g1:
    a = tf.Variable([1, 2, 3])
    
with tf.Graph().as_default() as g2:
    b = tf.Variable([1, 2, 3], name='var')
~~~

这两个图是不一样的，因为使用了不同的`name`。

~~~python
with tf.Graph().as_default() as g1:
    a = tf.Variable([1, 2, 3], name='a')
    b = tf.Variable([4, 5], name='b')
    
with tf.Graph().as_default() as g2:
    c = tf.Variable([4, 5], name='b')
    d = tf.Variable([1, 2, 3], name='a')
~~~

这两个图是一致的。

~~~python
with tf.Graph().as_default() as g1:
    a = tf.Variable([1, 2, 3])
    b = tf.Variable([4, 5])
    
with tf.Graph().as_default() as g2:
    c = tf.Variable([4, 5])
    d = tf.Variable([1, 2, 3])
~~~

这两个图是不一致。看起来，两个图一模一样，然而Tensorflow会给每个没有`name`的Op进行命名，两个图由于均使用了2个Variable，并且都没有主动命名，所以在`g1`中a的`name`与`g2`中c的`name`不同，`g1`中b的`name`与`g2`中d的`name`不同。



## 4. 变量的值固化到图中

当我们训练好了一个模型时，就意味着模型中的变量的值确定了，不在需要改变了。这时候如果我们想要将训练好的模型运用于生产，按照上述所讲，我们需要分别将图与变量保存（或者保存变量时也保存了meta），这是一种方法。但这种方法是为训练与调整模型所设计，为了还原变量，我们不得不在会话中操作图。例如，现在已经训练好了模型A，现需要利用模型A作为输入，训练模型B，按照上述还原模型的方法，必须在会话中还原模型，那么构建模型的流程就发生了变化，而我们希望算法的设计全部都在图中完成，这就使得图的设计变得麻烦。

所以为了使用简便，可以将所有变量固化成常量，随图一起保存。这样使用起来也更加简便，我们只需要导入图即可完成模型的完整导入。利用固化后的图参与构建新的图也变得容易了。

实现上述功能，需要操作GraphDef中的节点，好在Tensorflow已经为我们提供了相关的API：`convert_variables_to_constants`。

`convert_variables_to_constants`在`tensorflow.python.framework.graph_util`中，用法如下：

~~~python
# 返回一个新的图
convert_variables_to_constants(
    sess,  # 会话
    input_graph_def,  # 图定义
    output_node_names,  # 输出节点(不需要的节点在生成的新图将被排除)(注意是节点而不是Tensor)
    variable_names_whitelist=None,  # 将指定的variable转成constant
    variable_names_blacklist=None)  # 指定的variable不转成constant
~~~

例如：

将变量转化为常量，边存成图。

~~~python
import tensorflow as tf
from tensorflow.python.framework import graph_util

with tf.Graph().as_default() as g:
    my_input = tf.placeholder(dtype=tf.int32, shape=[], name='input')
    var = tf.get_variable(name='var', shape=[], dtype=tf.int32)
    output = tf.add(my_input, var, name='output')

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(var.assign(10))
    
    # 将变量转化为常量，返回一个新的图
    new_graph = graph_util.convert_variables_to_constants(
        sess, 
        sess.graph_def, 
        output_node_names=['output'])
    # 将新的图保存
    tf.train.write_graph(new_graph, '', 'graph.pb', as_text=False)
~~~

导入刚刚序列化的图：

~~~python
with tf.Graph().as_default() as new_graph:
    x = tf.placeholder(dtype=tf.int32, shape=[], name='x')
    with tf.gfile.FastGFile('graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_out = tf.import_graph_def(
            graph_def, 
            input_map={'input:0': x}, 
            return_elements=['output:0'])
        
with tf.Session(graph=new_graph) as sess:
    print(sess.run(g_out[0], feed_dict={x: 5}))  # >> 15
~~~

---

# 七.队列与线程

在Python中，线程和队列的使用范围很广，例如可以将运行时间较长的代码放在线程中执行，避免页面阻塞问题，例如利用多线程和队列快速读取数据等。然而在Tensorflow中，线程与队列的使用场景基本上是读取数据。并且**在1.2之后的版本中使用了`tf.contrib.data`模块代替了多线程与队列读取数据的方法（为了兼容旧程序，多线程与队列仍然保留）。**

## 1. 队列

Tensorflow中拥有队列的概念，用于操作数据，队列这种数据结构如在银行办理业务排队一样，队伍前面的人先办理，新进来的人排到队尾。队列本身也是图中的一个节点，与其相关的还有入队（enqueue）节点和出队（dequeue）节点。入队节点可以把新元素插入到队列末尾，出队节点可以把队列前面的元素返回并在队列中将其删除。

在Tensorflow中有两种队列，即FIFOQueue和RandomShuffleQueue，前者是标准的先进先出队列，后者是随机队列。

### 1.1 FIFOQueue

FIFOQueue创建一个先进先出队列，这是很有用的，当我们需要加载一些有序数据，例如按字加载一段话，这时候我们不能打乱样本顺序，就可以使用队列。

创建先进先出队列使用`ft.FIFOQueue()`，具体如下：

~~~python
tf.FIFOQueue(
    capacity,  # 容量 元素数量上限
    dtypes, 
    shapes=None, 
    names=None, 
    shared_name=None,
    name='fifo_queue')
~~~

如下，我们定义一个容量为3的，元素类型为整形的队列：

~~~python
queue = tf.FIFOQueue(3, tf.int32)
~~~

**注意**：`ft.FIFOQueue()`的参数`dtypes`、`shapes`、`names`均是列表（当列表的长度为1时，可以使用元素代替列表），且为对应关系，如果dtypes中包含两个dtype，则shapes中也包含两个shape，names也包含两个name。其作用是同时入队或出队多个有关联的元素。

如下，定义一个容量为3，每个队列值包含一个整数和一个浮点数的队列：

~~~python
queue = tf.FIFOQueue(
  	3,
  	dtypes=[tf.int32, tf.float32],
  	shapes=[[], []],
  	names=['my_int', 'my_float'])
~~~

队列创建完毕之后，就可以进行出队与入队操作了。

#### 入队

入队即给队列添加元素。使用`quene.enqueue()`或`queue.enqueue_many()`方法，前者用来入队一个元素，后者用来入队0个或多个元素。如下：

~~~python
queue = tf.FIFOQueue(3, dtypes=tf.float32)
# 入队一个元素 简写
eq1 = queue.enqueue(1.)
# 入队一个元素 完整写法
eq2 = queue.enqueue([1.])
# 入队多个元素 完整写法
eq3 = queue.enqueue_many([[1.], [2.]])

with tf.Session() as sess:
  	sess.run([eq1, eq2, eq3])
~~~

如果入队操作会导致元素数量大于队列容量，则入队操作会阻塞，直到出队操作使元素数量减少到容量值。

当我们指定了队列中元素的names时，我们在入队时需要使用使用字典来指定入队元素，如下：

~~~python
queue = tf.FIFOQueue(
    3, 
    dtypes=[tf.float32, tf.int32],
    shapes=[[], []],
    names=['my_float', 'my_int'])
queue.enqueue({'my_float': 1., 'my_int': 1})
~~~

#### 出队

出队即给从队列中拿出元素。出队操作类似于入队操作`queue.dequeue()`、`queue.dequeue_many()`分别出队一个或多个元素，使用。如下：

~~~python
queue = tf.FIFOQueue(3, dtypes=tf.float32)
queue.enqueue_many([[1.], [2.], [3.]])

val = queue.dequeue()
val2 = queue.dequeue_many(2)

with tf.Session() as sess:
  	sess.run([val, val2])
~~~

如果队列中元素的数量不够出队操作所需的数量，则出队操作会阻塞，知道入队操作加入了足够多的元素。

### 1.2 RandomShuffleQueue

RandomShuffleQueue创建一个随机队列，在执行出队操作时，将以随机的方式拿出元素。当我们需要打乱样本时，就可以使用这种方法。例如对小批量样本执行训练时，我们希望每次取到的小批量的样本都是随机组成的，这样训练的算法更准确，此时使用随机队列就很合适。

使用`tf.RandomShuffleQueue()`来创建随机队列，具体如下：

~~~python
tf.RandomShuffleQueue(
  	capacity,  # 容量
  	min_after_dequeue,  # 指定在出队操作之后最少的元素数量，来保证出队元素的随机性
  	dtypes, 
  	shapes=None, 
  	names=None, 
  	seed=None, 
  	shared_name=None, 
  	name='random_shuffle_queue')
~~~

RandomShuffleQueue的出队与入队操作与FIFOQueue一样。

举例：

~~~python
queue = tf.RandomShuffleQueue(10, 2, dtypes=tf.int16, seed=1)

with tf.Session() as sess:
    for i in range(10):
        sess.run(queue.enqueue([i]))
    print('queue size is : %d ' % sess.run(queue.size()))
    
    for i in range(8):
        print(sess.run(queue.dequeue()))
    print('queue size is : %d ' % sess.run(queue.size()))
    queue.close()
~~~

执行结果如下：

```
queue size is : 10 
7
8
1
3
9
4
2
6
queue size is : 2 
```

本例中用到了`queue.size()`来获取队列的长度，使用`queue.close()`来关闭队列。可以看到顺序入队，得到的是乱序的数据。



## 2. 线程

在Tensorflow1.2之前的版本中，数据的输入，需要使用队列配合多线程。在之后的版本中将不再推荐使用这个功能，而是使用`tf.contrib.data`模块代替。

Tensorflow中的线程调用的是Python的`threading`库。并增加了一些新的功能。

### 2.1 线程协调器

Python中使用多线程，会有一个问题，即多个线程运行时，一个线程只能在运行完毕之后关闭，如果我们在线程中执行了一个死循环，那么线程就不存在运行完毕的状态，那么我们就无法关闭线程。例如我们使用多个线程给队列循环入队数据，那么我们就无法结束这些线程。线程协调器可以用来管理线程，让所有执行的线程停止运行。

线程协调器使用类`tf.train.Coordinator()`来管理。

如下：

~~~python
import tensorflow as tf
import threading

# 将要在线程中执行的函数
# 传入一个Coordinator对象
def myloop(coord):
    while not coord.should_stop():
        ...do something...
        if ...some condition...:
            coord.request_stop()
# 创建Coordinator对象
coord = tf.train.Coordinator()
# 创建10个线程
threads = [threading.Thread(target=myLoop, args=(coord,)) for _ in range(10)]
# 开启10个线程
for t in threads:
  	t.start()
# 等待线程执行结束
coord.join(threads)
~~~

上述代码需要注意的是`myloop`函数中，根据`coord.should_stop()`的状态来决定是否运行循环体，默认情况下`coord.should_stop()`返回`False`。当某个线程中执行了`coord.request_stop()`之后，所有线程在执行循环时，都会因为`coord.should_stop()`返回`True`而终止执行，从而使所有线程结束。



### 2.2 队列管理器

上面队列的例子中，出队、入队均是在主线程中完成。这时候会有一个问题：当入队操作速度较慢，例如加载较大的数据时，入队操作不仅影响了主线程的总体速度，也会造成出队操作阻塞，更加拖累了线程的执行速度。我们希望入队操作能够在一些列新线程中执行，主进程仅仅执行出队和训练任务，这样就可以提高整体的运行速度。

如果直接使用Python中的线程进行管理Tensorflow的队列会出一些问题，例如子线程无法操作主线程的队列。在Tensorflow中，队列管理器可以用来创建、运行、管理队列线程。队列管理器可以管理一个或多个队列。

#### 单队列管理

用法如下：

1. 使用`tf.train.QueueRunner()`创建一个队列管理器。队列管理器中需要传入队列以及队列将要执行的操作和执行的线程数量。
2. 在会话中开启所有的队列管理器中的线程。
3. 使用Coordinator通知线程结束。

例如：

~~~python
g = tf.Graph()
with g.as_default():
    q = tf.FIFOQueue(100, tf.float32)
    
    enq_op = q.enqueue(tf.constant(1.))
    
    # 为队列q创建一个拥有两个线程的队列管理器
    num_threads = 2
    # 第一个参数必须是一个队列，第二个参数必须包含入队操作
    qr = tf.train.QueueRunner(q, [enq_op] * num_threads)

with tf.Session(graph=g) as sess:
    coord = tf.train.Coordinator()
    # 在会话中开启线程并使用coord协调线程
    threads = qr.create_threads(sess, start=True, coord=coord)
    
    deq_op = q.dequeue()
    for i in range(15):
        print(sess.run(deq_op))
    
    # 请求关闭线程
    coord.request_stop()
    coord.join(threads)
~~~

#### 多队列管理

队列管理器还可以管理多个队列，例如可以创建多个单一的队列管理器分别进行操作。但为了统一管理多个队列，简化代码复杂度，Tensorflow可以使用多队列管理。即创建多个队列管理器，统一开启线程和关闭队列。

使用方法如下：

1. 与单队列意义，使用`tf.train.QueueRunner()`创建一个队列管理器（也可以使用`tf.train.queue_runner.QueueRunner()`创建队列管理器，两个方法一模一样）。一个队列管理器管理一个队列。队列管理器中需要传入将要执行的操作以及执行的线程数量。例如：

   ~~~python
   # 队列1
   q1 = tf.FIFOQueue(100, dtypes=tf.float32)
   enq_op1 = q1.enqueue([1.])
   qr1 = tf.train.QueueRunner(q1, [enq_op1] * 4)

   # 队列2
   q2 = tf.FIFOQueue(50, dtypes=tf.float32)
   enq_op2 = q2.enqueue([2.])
   qr2 = tf.train.QueueRunner(q2, [enq_op2] * 3)
   ~~~

2. `tf.train.add_queue_runner()`向图的队列管理器集合中添加一个队列管理器（也可以使用`tf.train.queue_runner.add_queue_runner()`添加队列管理器，两个方法一模一样）。一个图可以有多个队列管理器。例如：

   ~~~python
   tf.train.add_queue_runner(qr1)
   tf.train.add_queue_runner(qr2)
   ~~~

3. `tf.train.start_queue_runners()`在会话中开启所有的队列管理器（也可以使用`tf.train.queue_runner.start_queue_runners()`开启，两个方法一模一样），即开启线程。这时候，我们可以将线程协调器传递给它，用于完成线程执行。例如：

   ~~~python
   with tf.Session() as sess:
       coord = tf.train.Coordinator()
       threads = tf.train.start_queue_runners(sess, coord=coord)
   ~~~

4. 任务完成，结束线程。使用Coordinator来完成：

   *注意：结束线程的这些操作还用到了Session，应该放在Session上下文中，否则可能出现Session关闭的错误。*

   ~~~python
   coord.request_stop()
   coord.join(threads)
   ~~~

完整案例：

在图中创建了一个队列，并且使用队列管理器进行管理。创建的队列

~~~python
g = tf.Graph()
with g.as_default():
    # 创建队列q1
    q1 = tf.FIFOQueue(10, tf.int32)
    enq_op1 = q1.enqueue(tf.constant(1))
    qr1 = tf.train.QueueRunner(q1, [enq_op1] * 3)
    tf.train.add_queue_runner(qr1)
    
    # 创建队列q2
    q2 = tf.FIFOQueue(10, tf.int32)
    enq_op2 = q2.enqueue(tf.constant(2))
    qr2 = tf.train.QueueRunner(q2, [enq_op2] * 3)
    tf.train.add_queue_runner(qr2)

with tf.Session(graph=g) as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coord)
    
    my_data1 = q1.dequeue()
    my_data2 = q2.dequeue()

    for _ in range(15):
        print(sess.run(my_data1))
        print(sess.run(my_data2))
    
    coord.request_stop()
    coord.join(threads)
~~~



#### 队列中op的执行顺序

队列管理器中会指定入队op，但有些时候入队op执行时，还需要同时执行一些别的操作，例如我们希望按顺序生成一个自然数队列，我们每次入队时，需要对入队的变量加1。然而如下写法输出的结果并没有按照顺序来：

~~~python
q = tf.FIFOQueue(100, tf.float32)
counter = tf.Variable(tf.constant(0.))

assign_op = tf.assign_add(counter, tf.constant(1.))
enq_op = q.enqueue(counter)

# 使用一个线程入队
qr = tf.train.QueueRunner(q, [assign_op, enq_op] * 1)

with tf.Session() as sess:
    qr.create_threads(sess, start=True)
    
    for i in range(10):
        print(sess.run(q.dequeue()))  
# 输出结果        
# 1.0
# 3.0
# 4.0
# 4.0
# 5.0
# 7.0
# 9.0
# 10.0
# 12.0
# 12.0
~~~

可以看到`assign_op`与`enq_op`的执行是异步的，入队操作并没有严格的在赋值操作之后执行。


---

# 八.数据集文件操作


在机器学习中，文件读取是最常见的也较复杂的操作之一。通常，我们训练一个模型或者部署运行一个模型均需要对文件进行读写，例如获取训练数据集与保存训练模型。保存模型之前我们已经讲过了，这一节主要讲对训练数据集的读取操作。

Tensorflow读取训练数据主要有以下三种方法：

* feeding：即每次运行算法时，从Python环境中供给数据。
* 常量：在图中设置一些常量进行保存数据。这仅仅适合数据量较小的情况。
* 文件读取。直接从文件中读取。

机器学习中训练模型使用的数据集通常比较大，有些数据往往存储在多个文件中，而另一些数据则根据其功能写入不同的文件中，这时候从文件中读取比较方便。例如MNIST中包含4个文件。如下：

~~~wiki
train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
~~~

第一个文件是MNIST的训练集中的图片，第二个文件是训练集中的标记，第三个文件是测试集中的图片，第四个文件是测试集中的标记。

这样的好处是数据集的数据结构清晰。但有时候这样也会加大我们获取数据的难度，例如我们必须同时读取不同的文件来获得一条完整的数据样本，例如我们需要设置复杂的多线程加快数据读取。而Tensorflow提供了读取数据的便捷方法。

## 1. 文件读取数据的流程

从文件中读取数据是最常见的数据读取的方法。这里的“文件”是指文本文件、二进制文件等。从文件中读取数据可以配合之前所学的队列与线程的知识，高效的完成工作。

这里我们首先介绍从文件中读取数据的方法，之后再去结合线程与队列进行使用。

### 1.1 文件读取

文件读取可以分为两种情况，第一种是直接读取文件，第二种是从文件名队列中读取数据。对于文件较多的数据集，我们通常采用第二种方法。

#### 1.1.1 直接读取文件

使用`tf.read_file`对文件进行读取，返回一个`string`类型的Tensor。

~~~python
tf.read_file(filename, name=None)
~~~

一个数据结构如下的csv文件，其读取的结果是：`b'1,11\n2,22\n3,33\n'`

~~~tex
1,11
2,22
3,33
~~~

可以看到，这种方法仅仅适合读取一个文件就是一个样本（或特征）的数据，例如图片。由于解码不方便，其不适合读取文本文件或其它文件。

#### 1.1.2 从文件名队列中读取

为什么我们需要从文件名队列中读取呢？因为随着样本数据规模的增大，文件的数量急剧增多，如何管理文件就成为了一个问题。例如，我们拥有100个文件，我们希望读取乱序读取文件生成5代（epoch）样本，直接管理文件就很麻烦。使用队列，就可以解决这个问题。

从文件名队列中读取文件有2个步骤。

##### 步骤一：生成文件名队列

Tensorflow为我们提供了文件名队列`tf.train.string_input_producer`用于管理被读取的文件。

~~~python
# 返回一个队列，同时将一个QueueRunner加入图的`QUEUE_RUNNER`集合中
tf.train.string_input_producer(
    string_tensor,  # 1维的string类型的文件名张量 可以是文件路径
    num_epochs=None,  # 队列的代，设置了之后，超出队列会出现OutOfRange错误，且需要使用local_variables_initializer()初始化
    shuffle=True,  # True代表每一个代中的文件名是打乱的
    seed=None,  # 随机种子
    capacity=32,  # 队列容量
    shared_name=None,
    name=None,
    cancel_op=None)
~~~

**注意：`local_variables_initializer()`需要写在用到local_variable的地方的后面。这与`global_variables_initializer()`用法不太一样。**

例如，生成两个csv文件名队列：

~~~python
filename_queue = tf.train.string_input_producer(['1.csv', '2.csv'])
~~~

或者我们可以使用列表推导来生成文件名队列，例如：

~~~python
filename_queue = tf.train.string_input_producer(['%d.csv' % i for i in range(1, 3)])
~~~

对于简单的、文件较少的数据集，使用上述方法生成文件名队列很合适。但面对大型的数据集，这样操作就不够灵活，例如中文语音语料库thchs30训练数据集的音频文件有一个目录A2中的文件如下：

![]( http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/thchs30-train-wav-v2.jpg)

这个数据集目录中包含类了似于A2这样的目录就有几十个，每个目录中都有200个左右的文件。这些文件的名称都有一定的规律，例如A2目录下的文件的文件名开头都是'A2'，结尾都是'.wav'中间是不连续的1-n的自然数。我们无法使用列表推导`['A2_%d.wav' % i for i in range(n)]`（因为不连续）。

Tensorflow提供了获取文件名的模式匹配的方法`tf.train.match_filenames_once`。可以灵活的使用此方法获取我们所要的文件名，过滤掉不需要的文件的文件名。用法如下：

```python
tf.match_filenames_once(
    pattern,  # 文件名匹配模式（glob，可以看做是简化版的正则表达式）或者一个文件名Tensor
    name=None
)
```

glob是Linux内建的用于文件路径查找的函数，glob通配符类型有：

- `*` ：任意长度的任意字符
- `?` ：任意单个字符
- `[]` ：匹配指定范围内的单个字符
- `[0-9]` ：单个数字
- `[a-z]` ：不区分大小写的a-z
- `[A-Z]` ：大写字符
- `[^]` ：匹配指定范围外的单个字符

使用`tf.train.match_filenames_once`获取文件名队列的代码如下：

```python
filenames = tf.match_filenames_once('A2_[0-9]*.wav')  # 这里也可以直接写 '*'
```

然后可以根据所有文件名`filenames`生成文件名队列：

~~~python
filename_queue = tf.train.string_input_producer(filenames)
~~~

##### 步骤二：根据文件名队列读取相应文件

文件名队列出队得到的是文件名（也可以是文件路径），读取数据正是从这些文件名中找到文件并进行数据读取的。但需要注意的是，Tensorflow实现了配合“文件名出队”操作的数据读取op，所以我们并不需要写出队操作与具体的文件读取方法。同时读取数据的方法也有很多种，这里我们以读取csv文件为例，下文中，我们给出了更多种文件的读取方法。

读取csv文件，需要使用**文本文件读取器**`tf.TextLineReader`，与csv解码器一起工作。

[CSV文件](https://tools.ietf.org/html/rfc4180)就是以逗号进行分割值，以\\n为换行符的文本文件，文件的后缀名为.csv。

`tf.TextLineReader`用法如下：

```python
# 创建一个读取器对象
# 输出由换行符分割的文件读取器 可以用来读取csv、text等文本文件
tf.TextLineReader(skip_header_lines=None, name=None)

# 从文件名队列中读取数据 并返回下一个`record`(key, value)
# 相当于出队操作
tf.TextLineReader.read(filename_queue, name=None)
```

读取数据之后，我们还需要对数据进行解码，将其转变为张量，CSV文件的解码方法如下：

```python
# 将csv文件解码成为Tensor
tf.decode_csv(
  	records,
  	record_defaults,  # 默认值，当 records 中有缺失值时，使用默认值填充
  	field_delim=',',  # 分隔符
  	use_quote_delim=True,  # 当为True时，会去掉元素的引号，为False时，会保留元素的引号
  	name=None)
```



继续步骤一中未完成的例子，读取数据的操作如下：

~~~python
reader = tf.TextLineReader.read()
key, value = reader.read(filename_queue)  # filename_queue是第一步生成的文件名队列

decoded = tf.decode_csv(value, record_defaults)  # record_defaults是必填项，此处省略了。
~~~

这里得到的key是文件名与行索引，value是key对应的张量值。key与value均为`tf.string`类型。

到这里，我们已经通过文件名队列获取到了数据，完成了我们的目标。



### 1.2 利用队列存取数据

虽然我们已经读取并解码了文件，但直接使用解码后的张量依然有很多的问题。例如我们希望使用“批量样本”，而其产生的是单个样本；例如当生成样本的速度与消费样本的速度不匹配时，可能造成阻塞。这些问题，我们可以使用队列来解决。

上一章，我们使用了队列对张量数据进行存储与读取。使用队列最大的优点就是可以充分发挥生产者-消费者的模式。而上文中，我们已经把文件读取并转换为了张量，那么我们就可以使用队列来管理读取的数据。

我们仍以读取csv文件为例进行演示：

~~~python
g = tf.Graph()
with g.as_default():
    # 生成“文件名队列”
    filenames = tf.train.match_filenames_once('./csv/*.csv')
    filename_queue = tf.train.string_input_producer(filenames)

    # 读取数据     
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # 解码csv文件  record_default的值根据实际情况写
    decoded = tf.decode_csv(value, record_defaults=[[0], [0]])

    # 创建“样本队列”  这里容量与类型需要根据实际情况填写 
    example_queue = tf.FIFOQueue(5, tf.int32)
    # 入队操作    
    enqueue_example = example_queue.enqueue([decoded])
    # 出队操作   根据需要也可以dequeue_many 
    dequeue_example = example_queue.dequeue()

    # 创建队列管理器  根据需要制定线程数量num_threads，这里为1
    qr = tf.train.QueueRunner(example_queue, [enqueue_example] * 1)
    # 将qr加入图的QueueRunner集合中
    tf.train.add_queue_runner(qr)

with tf.Session(graph=g) as sess:
    # 创建线程协调器
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 出队数据
    for i in range(15):
        print(sess.run(dequeue_example))

    # 清理
    coord.request_stop()
    coord.join(threads)
~~~

上面的代码就是完整的从文件中读取数据，并存在队列中的例子。这个例子中有3处使用了队列，分别是：

* 文件名队列
* 数据读取队列
* 样本队列

通过使用队列，使得数据的读取与使用变得简便与条理。但可以看到上面的代码仍然较繁琐，Tensorflow提供了更加简便的API来简化操作，和完善功能。

### 1.3 简便的批量样本生成

上面的读取文件的例子中，使用了3个队列，其中“文件名队列”与“样本数据读取队列”已经被Tensorflow抽象成了简便的API，即我们不需要显式的创建队列和创建对应的队列管理器。其实第三个队列“样本队列”，Tensorflow也给出了更加简便的API，这些API的用法如下：

注意：以下几个API均可以完成样本生成，但功能略微不同。

* `tf.train.batch`

  API的详情如下：

  ~~~python
  # 返回一个dequeue的OP，这个OP生成batch_size大小的tensors
  tf.train.batch(
      tensors,  # 用于入队的张量
    	batch_size,  # 出队的大小
    	num_threads=1,  # 用于入队的线程数量
    	capacity=32,  # 队列容量
    	enqueue_many=False,  # 入队的样本是否为多个
    	shapes=None,  # 每个样本的shapes 默认从tensors中推断
    	dynamic_pad=False,  # 填充一个批次中的样本，使之shape全部相同
    	allow_smaller_final_batch=False,  # 是否允许最后一个批次的样本数量比正常的少
    	shared_name=None,
    	name=None)
  ~~~

  在读取数据并解码之后，可以将张量送入此方法。

  读取文件时，只需要将上面的例子中的批量样本生成代码替换为此即可，例如：

  ~~~python
  ...
  decoded = tf.decode_csv(...)

  dequeue_example = tf.train.batch(decoded, batch_size=5)
  ~~~

  比之前少些了4行代码。更加简洁明了。

  ​

* `tf.train.batch_join`

  API的详情如下：

  ~~~python
  # 返回一个dequeue的OP，这个OP生成batch_size大小的tensors
  tf.train.batch_join(
      tensors_list,  # 入队
      batch_size,
      capacity=32,
      enqueue_many=False,
      shapes=None,
      dynamic_pad=False,
      allow_smaller_final_batch=False,
      shared_name=None,
      name=None)
  ~~~

  此方法与上述方法类似，但是创建多线程的方法不同。这里是根据tensors_list的长度来决定线程的数量的。用法如下：

  ~~~python
  ...
  decoded_list = [tf.decode_csv(...) for _ in range(2)]

  dequeue_example = tf.train.batch_join(decoded_list, batch_size=5)
  ~~~

  这里创建了2个解码器，在数据读取时，会分别给这两个解码op创建各自的线程。本质上与上面的方法是一样。

* `tf.train.shuffle_batch`

  与`tf.train.batch`相比，此方法可以打乱输入样本。API详情如下：

  ~~~python
  tf.train.shuffle_batch(
      tensors, 
      batch_size, 
      capacity, 
      min_after_dequeue,
      num_threads=1,
      seed=None,  # 随机数种子
      enqueue_many=False, 
      shapes=None,
      allow_smaller_final_batch=False, 
      shared_name=None, 
      name=None)
  ~~~

  用法与`tf.train.batch`类似。

* `tf.train.shuffle_batch_join`

  与`tf.train.batch_join`相比，此方法可以打乱输入样本。API详情如下：

  ~~~python
  tf.train.shuffle_batch(
      tensors, 
      batch_size, 
      capacity, 
      min_after_dequeue,
      num_threads=1, 
      seed=None, 
      enqueue_many=False, 
      shapes=None,
      allow_smaller_final_batch=False, 
      shared_name=None, 
      name=None)
  ~~~

  用法与`tf.train.batch_join`类似。


### 1.4 完整读取数据一般步骤

从文件中读取数据的一般步骤如下：

1. 生成文件名队列。
2. 读取文件，生成样本队列。
3. 从样本队列中生成批量样本。

![]( http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/AnimatedFileQueues.gif)



例子：

~~~python
# 第一步 生成文件名队列
filename_queue = tf.train.string_input_producer(['1.csv', '2.csv'])

# 第二步 根据文件名读取数据
reader = tf.TextLineReader(filename_queue)
record = reader.read(filename_queue)
# 解码数据成为Tensor
csv_tensor = tf.decode_csv(record, default_record=[...])

# 第三步 根据Tensors生产批量样本
batch_example = tf.train.batch(csv_tensor, batch_size)
~~~



## 2. 高效读取数据

上面的例子，我们直接从原始文件中读取数据。当文件数量较多，且解码不够迅速时，上面的方法就显示出了一些缺陷。在Tensorflow中，我们可以使用TFRecord对数据进行存取。TFRecord是一种二进制文件。可以更快速的操作文件。

通常我们得到的数据集并不是TFRecord格式，例如MNIST数据集也是一个二进制文件，每一个字节都代表一个像素值（除去开始的几个字节外）或标记，这与TFRecord文件的数据表示方法（TFRecord的二进制数据中还包含了校验值等数据）并不一样。所以，通常我们需要将数据转化为TFRecord文件。这里需要注意并不是每一个数据集均需要转化为TFRecord文件，建议将文件数量较多，直接读取效率低下的数据集转化为TFRecord文件格式。

### 2.1 写入TFRecord文件

TFRecord文件的存取，本质上是对生成的包含样本数据的ProtoBuf数据的存取，TFRecord文件只适合用来存储数据样本。一个TFRecord文件存储了一个或多个example对象，example.proto文件描述了一个样本数据遵循的格式。每个样本example包含了多个特征feature，feature.proto文件描述了特征数据遵循的格式。

在了解如何写入TFRecord文件前，我们首先了解一下其对应的消息定义文件。通过这个文件，我们可以知道消息的格式。

**feature.proto**的内容如下（删除了大部分注释内容）：

~~~protobuf
syntax = "proto3";
option cc_enable_arenas = true;
option java_outer_classname = "FeatureProtos";
option java_multiple_files = true;
option java_package = "org.tensorflow.example";

package tensorflow;

// Containers to hold repeated fundamental values.
message BytesList {
  repeated bytes value = 1;
}
message FloatList {
  repeated float value = 1 [packed = true];
}
message Int64List {
  repeated int64 value = 1 [packed = true];
}

// Containers for non-sequential data.
message Feature {
  // Each feature can be exactly one kind.
  oneof kind {
    BytesList bytes_list = 1;
    FloatList float_list = 2;
    Int64List int64_list = 3;
  }
};

message Features {
  // Map from feature name to feature.
  map<string, Feature> feature = 1;
};

message FeatureList {
  repeated Feature feature = 1;
};

message FeatureLists {
  // Map from feature name to feature list.
  map<string, FeatureList> feature_list = 1;
};
~~~

可以看到一个特征`Feature`可以是3中数据类型（`BytesList`，`FloatList`，`Int64List`）之一。多个特征`Feature`组成一个组合特征`Features`，多个组合特征`Features`组成特征列表`FeatureList`。多个特征列表组成特征列表组`FeatureLists`。

**example.proto**的内容如下（删除了大部分注释内容）：

```protobuf
syntax = "proto3";

import "tensorflow/core/example/feature.proto";
option cc_enable_arenas = true;
option java_outer_classname = "ExampleProtos";
option java_multiple_files = true;
option java_package = "org.tensorflow.example";

package tensorflow;

message Example {
  Features features = 1;
};

message SequenceExample {
  Features context = 1;
  FeatureLists feature_lists = 2;
};
```

可以看到一个样本`Example`包含一个特征组合。序列样本`SequenceExample`包含一个类型是特征组合的上下文`context`与一个特征列表组`feature_lists`。

可以看到：**TFRecord存储的样本数据是以样本为单位的。**

了解了TFRecord读写样本的数据结构之后，我们就可以使用相关API进行操作。

#### 写入数据

Tensorflow已经为我们封装好了操作protobuf的方法以及文件写入的方法。写入数据的第一步是打开文件并创建writer对象，Tensorflow使用`tf.python_io.TFRecordWriter`来完成，具体如下：

~~~python
# 传入一个路径，返回一个writer上下文管理器
tf.python_io.TFRecordWriter(path, options=None)
~~~

TFRecordWriter拥有`write`，`flush`，`close`方法，分别用于写入数据到缓冲区，将缓冲区数据写入文件并清空缓冲区，关闭文件流。

开启文件之后，需要创建样本对象并将example数据写入。根据上面的proto中定义的数据结构，我们知道一个样本对象包含多个特征。所以我们首先需要创建特征对象，然后再创建样本对象。如下为序列化一个样本的例子：

~~~python
with tf.python_io.TFRecordWriter('./test.tfrecord') as writer:
    f1 = tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
    f2 = tf.train.Feature(float_list=tf.train.FloatList(value=[1. , 2.]))
    b = np.ones([i]).tobytes()  # 此处默认为float64类型
    f3 = tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))

    features = tf.train.Features(feature={'f1': f1, 'f2': f2, 'f3': f3})
    example = tf.train.Example(features=features)

    writer.write(example.SerializeToString())
~~~

序列化多个样本只需要重复上述的写入过程即可。如下：

~~~python
with tf.python_io.TFRecordWriter('./test.tfrecord') as writer:
    # 多个样本多次写入
    for i in range(1, 6):
        f1 = tf.train.Feature(int64_list=tf.train.Int64List(value=[i]))
        f2 = tf.train.Feature(float_list=tf.train.FloatList(value=[1. , 2.]))
        b = np.ones([i]).tobytes()
        f3 = tf.train.Feature(bytes_list=tf.train.BytesList(value=[b]))
        
        features = tf.train.Features(feature={'f1': f1, 'f2': f2, 'f3': f3})
        example = tf.train.Example(features=features)
        
        writer.write(example.SerializeToString())
~~~

**注意事项：**

* `tf.train.Int64List`、`tf.train.FloatList`、`tf.train.BytesList`均要求输入的是python中的list类型的数据，而且list中的元素分别只能是int、float、bytes这三种类型。
* 由于生成protobuf数据对象的类中，只接受关键字参数，所以参数必须写出参数名。
* protobuf数据对象类需要遵循proto文件中定义的数据结构来使用。



TFRecord文件的数据写入是在Python环境中完成的，不需要启动会话。写入数据的过程可以看做是原结构数据转换为python数据结构，再转换为proto数据结构的过程。完整的数据写入过程如下：

1. 读取文件中的数据。
2. 组织读取到的数据，使其成为以“样本”为单位的数据结构。
3. 将“样本”数据转化为Python中的数据结构格式（int64\_list，float\_list，bytes\_list三种之一）。
4. 将转化后的数据按照proto文件定义的格式写出“Example”对象。
5. 将“Example”对象中存储的数据序列化成为二进制的数据。
6. 将二进制数据存储在TFRecord文件中。



### 2.2 读取TFRecord文件

读取TFRecord文件类似于读取csv文件。只不过使用的是`tf.TFRecordReader`进行读取。除此以外，还需要对读取到的数据进行解码。`tf.TFRecordReader`用法如下：

~~~python
reader = tf.TFRecordReader(name=None, options=None)
key, value = reader.read(queue, name=None)
~~~

此处读取到的value是序列化之后的proto样本数据，我们还需要对数据进行解析，这里可以使用方法`tf.parse_single_example`。解析同样需要说明解析的数据格式，这里可以使用`tf.FixedLenFeature`与`tf.VarLenFeature`进行描述。

解析单个样本的方法如下：

~~~python
# 解析单个样本
tf.parse_single_example(serialized, features, name=None, example_names=None)
~~~

解析设置数据格式的方法如下

~~~python
# 定长样本
tf.FixedLenFeature(shape, dtype, default_value=None)
# 不定长样本
tf.VarLenFeature(dtype)
~~~

完整例子如下（解析的是上文中写入TFRecord文件的例子中生成的文件）：

~~~python
reader = tf.TFRecordReader()
key, value = reader.read(filename_queue)

example = tf.parse_single_example(value, features={
    'f1': tf.FixedLenFeature([], tf.int64),
    'f2': tf.FixedLenFeature([2], tf.float32),
    'f3': tf.FixedLenFeature([], tf.string)})

feature_1 = example['f1']
feature_2 = example['f2']
feature_3 = tf.decode_raw(example['f3'], out_type=tf.float64)
~~~

这里还用到了`tf.decode_raw`用来解析bytes数据，其输出type应该等于输入时的type，否则解析出的数据会有问题。

```python
tf.decode_raw(
    bytes,  # string类型的tensor
    out_type,  # 输出类型，可以是`tf.half, tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.int64`
    little_endian=True,  # 字节顺序是否为小端序
    name=None)
```



无论使用哪一数据读取的方法，其过程都是一样的。过程如下：

1. 打开文件，读取数据流。
2. 将数据流解码成为指定格式。在TFRecord中，需要首先从proto数据解码成包含Example字典数据，再把其中bytes类型的数据解析成对应的张量。其它的比如csv则可以直接解析成张量。
3. 关闭文件。



## 3. 数据读取的更多方法

上文介绍了数据读取的流程以及完整读取数据的方法，但我们还可以更灵活的使用这些方法。同时，Tensorflow也提供了更多的方法来满足我们多种多样的需求。

### 3.1 多种文件读取器

上文中，我们介绍了从CSV文件中读取数据与TFRecord文件的读写数据的方式。事实中，Tensorflow除了可以读取CSV文件以及TFRecord文件以外，还可以读取二进制文件（非TFRecord文件）等多种文件。这里我们列出了Tensorflow支持的所有文件读取器。

* **`tf.ReaderBase`**

  所有文件读取器的基类。常用方法如下：

  * `tf.ReaderBase.read(queue, name=None)`：返回下一个记录 (key, value)。key表示其所读取的文件名以及行数等信息。value表示读取到的值，类型为bytes。
  * `tf.ReaderBase.read_up_to(queue, num_records, name=None)`：返回下num_records个记录(key, value)。其有可能返回少于num_records的记录。
  * `tf.ReaderBase.reset(name=None)`：将读取器恢复到初始状态。

* **`tf.TextLineReader`**

  文本文件读取器。可以用于读取csv、txt等文件。

* **`tf.WholeFileReader`**

  整个文件读取器。不同于TextLineReader读取到的数据是一行一行（以\\n换行符分割）的，WholeFileReader可以用于读取整个文件。例如读取一张图片时，我们可以使用这个方法进行读取。**仅读取一个文件时，可以使用`tf.read_file`**方法进行代替。

* **`tf.IdentityReader`**

  用于读取文件名。

* **`tf.TFRecordReader`**

  从TFRecord文件中读取数据。

* **`tf.FixedLengthRecordReader`**

  从文件中读取固定字节长度的数据。可以使用`tf.decode_raw`解码数据。

* **`tf.LMDBReader`**

  从LMDB数据库文件中读取。


---

# 九.TensorBoard基础用法


TensorBoard是Tensorflow自带的一个强大的可视化工具，也是一个web应用程序套件。在众多机器学习库中，Tensorflow是目前唯一自带可视化工具的库，这也是Tensorflow的一个优点。学会使用TensorBoard，将可以帮助我们构建复杂模型。

这里需要理解“可视化”的意义。“可视化”也叫做数据可视化。是关于数据之视觉表现形式的研究。这种数据的视觉表现形式被定义为一种以某种概要形式抽提出来的信息，包括相应信息单位的各种属性和变量。例如我们需要可视化算法运行的错误率，那么我们可以取算法每次训练的错误率，绘制成折线图或曲线图，来表达训练过程中错误率的变化。可视化的方法有很多种。但无论哪一种，均是对数据进行摘要(summary)与处理。

通常使用TensorBoard有三个步骤，首先需要在需要可视化的相关部位添加可视化代码，即创建摘要、添加摘要；其次运行代码，可以生成了一个或多个事件文件；最后启动TensorBoard的Web服务器。完成以上三个步骤，就可以在浏览器中可视化结果，Web服务器将会分析这个事件文件中的内容，并在浏览器中将结果绘制出来。

如果我们已经拥有了一个事件文件，也可以直接利用TensorBoard查看这个事件文件中的摘要。

##创建事件文件

在使用TensorBoard的第一个步骤中是添加可视化代码，即创建一个事件文件。创建事件文件需要使用`tf.summary.FileWriter()`。具体如下：

```python
tf.summary.FileWriter(
  	logdir,  # 事件文件目录
  	graph=None,  # `Graph`对象
  	max_queue=10,  # 待处理事件和摘要的队列大小
  	flush_secs=120, # 多少秒将待处理的事件写入磁盘
  	graph_def=None,  # [弃用参数]使用graph代替
  	filename_suffix=None)  # 事件文件的后缀名
```

`tf.summary.FileWriter()`类提供了一个在指定目录创建了一个事件文件(event files)并向其中添加摘要和事件的机制。该类异步地更新文件内容，允许训练程序调用方法来直接从训练循环向文件添加数据，而不会减慢训练，即既可以**在训练的同时跟新事件文件以供TensorBoard实时可视化**。

例如：

```python
import tensorflow as tf

writer = tf.summary.FileWriter('./graphs', filename_suffix='.file')
writer.close()
```

运行上述代码将会在`'./graphs'`目录下生成一个后缀名是`'.file'`的文件，这个文件正是事件文件。

**注意**：由于这个`writer`没有做任何操作，所以这是一个空的事件文件。如果不写`filename_suffix='.file'`这个属性，那么默认不创建空的事件文件。这时候如果使用TensorBoard的web服务器运行这个事件文件，看不到可视化了什么，因为这是一个空的事件文件。

##TensorBoard Web服务器

TensorBoard通过运行一个本地Web服务器，来监听6006端口。当浏览器发出请求时，分析训练时记录的数据，绘制训练过程中的图像。

TensorBoard的Web服务器启动的命令是：`tensorboard`，主要的参数如下：

~~~shell
--logdir  # 指定一个或多个目录。tensorboard递归查找其中的事件文件。
--host  # 设置可以访问tensorboard服务的主机
--port  # 设置tensorboard服务的端口号
--purge_orphaned_data  # 是否清除由于TensorBoard重新启动而可能已被修改的数据。禁用purge_orphaned_data可用于调试修改数据。
--nopurge_orphaned_data  
--reload_interval  # 后端加载新产生的数据的间隔
~~~

假如现在当前目录下`graphs`文件夹中存在事件文件，可以使用如下命令打开tensorboard web服务器：

~~~shell
tensorboard --logdir=./graphs
~~~

然后打开浏览器，进入`localhost:6006`即可查看到可视化的结果。

##可视化面板与操作详述

在Tensorflow中，summary模块用于可视化相关操作。TensorBoard目前支持8中可视化，即SCALARS、IMAGES、AUDIO、GRAPHS、DISTRIBUTIONS、HISTOGRAMS、EMBEDDINGS、TEXT。这8中可视化的主要功能如下。

- SCALARS：展示算法训练过程中的参数、准确率、代价等的变化情况。
- IMAGES：展示训练过程中记录的图像。
- AUDIO：展示训练过程中记录的音频。
- GRAPHS：展示模型的数据流图，以及训练在各个设备上消耗的内存与时间。
- DISTRIBUTIONS：展示训练过程中记录的数据的分布图。
- HISTOGRAMS：展示训练过程中记录的数据的柱状图。
- EMBEDDINGS：展示词向量（如Word2Vec）的投影分布。
- TEXT：

### GRAPHS

TensorBoard可以查看我们构建的数据流图，这对于复杂的模型的构建有很大的帮助。

####创建graph摘要

将数据流图可视化，需要将图传入summary的writer对象中。即在初始化`tf.summary.FileWriter()`时，给`graph`参数传入一个图。也可以在writer对象初始化之后使用`tf.add_graph()`给writer添加一个图。

如下，我们构建一个简单的图，并可视化：

~~~python
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(x))

writer.close()
~~~

在TensorBoard的GRAPHS中显示如下：

![]( http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/4.png)

图中`Const`、`Const_1`、`Add`表示的是节点的名称。

小圆圈表示为一个常量(constant)。椭圆形表示一个节点(OpNode)。箭头是一个引用边(reference edge)。当我们给代码中的op重命名之后，

~~~python
import tensorflow as tf

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(x))

writer.close()  # 注意关闭fileWriter
~~~

在TensorBoard中显示如下：

![]( http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/4.png)



点击一个节点，可以查看节点的详情，如下查看add节点的详情：

![]( http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/6.png)



图中不仅仅有节点和箭头，tensorflow有很多的用于图可视化的图形。所有的图像包括命名空间、op节点、没有连接关系的节点组、有连接关系的节点组、常量、摘要、数据流边、控制依赖边、引用边。具体如下：

![]( http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/graph_icon.png)

#### Unconnected Series

一系列的不相连接的操作相同且name相同（不包括系统给加的后缀）节点可以组成一个没有连接关系的节点组。

~~~python
import tensorflow as tf

with tf.Graph().as_default() as graph:
    a = tf.constant(2, name='a')
    b = tf.constant(3, name='b')
    x = tf.add(a, b, name='add')
    y = tf.add(a, b, name='add')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs')
    writer.add_graph(graph)
writer.close()
~~~

可视化之后如下：

![]( http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/graph_1.png)

这两个子图可以组成节点组，如下：

![]( http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/unconnected_series.png)

####Connected Series

一系列的相连接的操作相同且name相同（不包括系统给加的后缀）节点可以组成一个有连接关系的节点组。

~~~python
import tensorflow as tf

with tf.Graph().as_default() as graph:
    a = tf.constant(2, name='a')
    b = tf.constant(3, name='b')
    x = tf.add(a, b, name='add')
    y = tf.add(a, x, name='add')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs')
    writer.add_graph(graph)
writer.close()
~~~

可视化之后如下：

<img src=' http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/graph_2.png' width='300px'>

这两个子图可以组成节点组，如下：

<img src=' http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/connected_series.png' width='300px'>

#### Dataflow edge

~~~python
with tf.Graph().as_default() as graph:
    a = tf.constant(2, name='a')
    b = tf.constant(3, name='b')
    x = tf.add(a, b, name='add')
    y = tf.subtract(x, x, name='sub')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs')
    writer.add_graph(graph)
writer.close()
~~~



<img src=' http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/dataflow.png' width='270px'>



####Control Dependency edge

~~~python
with tf.Graph().as_default() as graph:
    a = tf.constant(2, name='a')
    b = tf.constant(3, name='b')
    x = tf.add(a, b, name='add')
    with graph.control_dependencies([x]):
        y = tf.subtract(a, b, name='sub')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs')
    writer.add_graph(graph)
writer.close()
~~~

<img src=' http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/contral_dependency.png' width='270px'>

#### 其它

在图可视化面板中，还有有多常用的功能，比如：切换不同的图；查看某次迭代时运行的内存、时间；标注节点使用的计算设备；导入别的图等。

###SCALARS

SCALARS面板可以展示训练过程中某些纯量数据的变化情况，例如模型参数的跟新情况、模型的正确率等。

使用scalars面板，主要是使用`tf.summary.scalar()`函数对张量进行采样，并返回一个包含摘要的protobuf的类型为string的0阶张量，函数的用法如下：

~~~python
# name 节点名称，也将作为在SCALARS面板中显示的名称
# tensor 包含单个值的实数数字Tensor
tf.summary.scalar(name, tensor, collections=None)
~~~

`tf.summary.scalar()`可以提取纯量摘要，提取到的摘要需要添加到事件文件中，这时候可以使用`tf.summary.FileWriter.add_summary()`方法添加到事件文件中。`add_summary()`的用法如下：

~~~python
add_summary(summary, global_step=None)
~~~

第一个参数是protobuf的类型为string的0阶张量摘要，第二个参数是记录摘要值的步数。例如训练100次模型，每次取一个摘要，这时候第二个参数就是第n次，这样可以得到纯量随训练变化的情况。

**注意**：`tf.summary.scalar()`获取到的并不是直接可以添加到事件文件中的数据，而是一个张量对象，必须在会话中执行了张量对象之后才能得到可以添加到事件文件中的数据。

完整创建纯量摘要的代码如下：

~~~python

import tensorflow as tf

with tf.Graph().as_default() as graph:
    var = tf.Variable(0., name='var')
    summary_op = tf.summary.scalar('value', var)  # 给变量var创建摘要
    
    random_val = tf.random_normal([], mean=0.5, stddev=1.)
    assign_op = var.assign_add(random_val)  # var累加一个随机数
    
with tf.Session(graph=graph) as sess:
  	# 创建一个事件文件管理对象
    writer = tf.summary.FileWriter('./graphs')  
    writer.add_graph(graph)
    
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        # 得到summary_op的值
        summary, _ = sess.run([summary_op, assign_op]) 
        # summary_op的值写入事件文件
        writer.add_summary(summary, i)
       
writer.close()
~~~

上述代码执行之后，在SCALARS面板中显示如下：

<img src=' http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/scalars.png'>

面板的右侧，可以看到一个曲线图，这个曲线图是我们创建的变量`var`的值的变化的曲线。纵坐标是我们设置的`summary_op`的`name`。曲线图可以放大、缩小，也可以下载。

面板左侧可以调整右侧的显示内容。我们看到的右侧的线是一个曲线，然而我们得到应该是一个折线图，这是因为左侧设置了平滑度(Smoothing)。横坐标默认是步数(STEP)，也可以设置为按照相对值(RELATIVE)或按照时间顺序(WALL)显示。

SCALARS面板也可以同时显示多个纯量的摘要。

###HISTOGRAMS

HISTOGRAMS面板显示的是柱状图，对于2维、3维等数据，使用柱状图可以更好的展示出其规律。例如

~~~python
import tensorflow as tf

with tf.Graph().as_default() as graph:
    var = tf.Variable(tf.random_normal([200, 10], mean=0., stddev=1.), name='var')
    histogram = tf.summary.histogram('histogram', var)   
    
with tf.Session(graph=graph) as sess:
    writer = tf.summary.FileWriter('./graphs')
    writer.add_graph(graph)
    
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        summaries, _ = sess.run([histogram, var])
        writer.add_summary(summaries, i)
       
writer.close()
~~~

上述代码，我们使用了`tf.summary.histogram()`方法建立了柱状图摘要，其使用方法类似于scalar摘要。在HISTOGRAMS面板中显示如下。

![]( http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/histograms.png)

### DISTRIBUTIONS

DISTRIBUTIONS面板与HISTOGRAMS面板相关联，当我们创建了柱状图摘要之后，在DISTRIBUTIONS面板也可以看到图像，上述HISTOGRAMS面板对应的在DISTRIBUTIONS面板中显示如下：

![]( http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/distributions.png)

DISTRIBUTIONS面板可以看做HISTOGRAMS面板的压缩后的图像。

### IMAGES

IMAGES面板用来展示训练过程中的图像，例如我们需要对比原始图像与算法处理过的图像的差异时，我们可以使用IMAGES面板将其展示出来。

建立IMAGES摘要的函数是：`tf.summary.image()`，具体如下：

~~~python
tf.summary.image(name, tensor, max_outputs=3, collections=None)
~~~

这里的参数`tensor`必须是4D的shape=`[batch_size,  height, width, channels]`的图片。参数`max_outputs`代表从一个批次中摘要几个图片。

举例，我们使用随机数初始化10张rgba的图片，并选取其中的前5张作为摘要：

~~~python
import tensorflow as tf

with tf.Graph().as_default() as graph:
    var = tf.Variable(tf.random_normal([10, 28, 28, 4], mean=0., stddev=1.), name='var')
    image = tf.summary.image('image', var, 5)   
    
with tf.Session(graph=graph) as sess:
    writer = tf.summary.FileWriter('./graphs')
    writer.add_graph(graph)
    
    sess.run(tf.global_variables_initializer())
    summaries, _ = sess.run([image, var])
    writer.add_summary(summaries)
    
writer.close()
~~~

在IMAGES面板中显示如下：

![]( http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/image.png)

### AUDIO

AUDIO面板与IMAGES面板类似，只不过是将图像换成了音频。音频数据比图像数据少一个维度。创建AUDIO摘要的方法是：`tf.summary.audio()`，具体如下：

~~~python
tf.summary.audio(name, tensor, sample_rate, max_outputs=3, collections=None)
~~~

这里的参数`tensor`必须是3D的shape=`[batch_size,  frames, channels]`或2D的shape=`[batch_size,  frames]`的音频。`sample_rate`参数。参数`max_outputs`代表从一个批次中摘要几个音频。

下面我们生成10段采样率为22050的长度为2秒的噪声，并选取其中的5个进行采样：

~~~python
import tensorflow as tf

with tf.Graph().as_default() as graph:
    var = tf.Variable(tf.random_normal([10, 44100, 2], mean=0., stddev=1.), name='var')
    audio = tf.summary.audio('video', var, 22050, 5)   
    
with tf.Session(graph=graph) as sess:
    writer = tf.summary.FileWriter('./graphs')
    writer.add_graph(graph)
    
    sess.run(tf.global_variables_initializer())
    summaries, _ = sess.run([audio, var])
    writer.add_summary(summaries)
    
writer.close()
~~~

在AUDIO面板中可以播放、下载采样的音频，显示效果如下：

![]( http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/audio.png)

###TEXT

TEXT面板用来可视化多行字符串，使用`tf.summary.text()`方法进行。

~~~python
tf.summary.text(name, tensor, collections=None)
~~~

举例：

~~~python
import tensorflow as tf

with tf.Graph().as_default() as graph:
    var = tf.constant(['Tensorflow 基础教程做的挺不错 中午加肉',
                       'Tensorboard 做的也很棒',
                       '你是一个全能型选手'])
    text = tf.summary.text('text', var)   
    
with tf.Session(graph=graph) as sess:
    writer = tf.summary.FileWriter('./graphs')
    writer.add_graph(graph)
    
    sess.run(tf.global_variables_initializer())
    summaries, _ = sess.run([text, var])
    writer.add_summary(summaries)

writer.close()
~~~

可视化效果如下：

![]( http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/text.png)

### EMBEDDINGS

EMBEDDINGS面板一般用来可视化词向量(word embedding)。在自然语言处理、推荐系统中较为常见。embeding的可视化与其它类型的可视化完全不同，需要用到tensorboard插件并配合检查点操作。具体教程我们在之后的内容中详述，这里，我们可以查看一下手写数字可视化后的效果：

![]( http://gytblog.oss-cn-shenzhen.aliyuncs.com/blog/tensorflow/course/embeddings.png)


----

## 来源

[资料来源](https://github.com/edu2act/course-tensorflow)