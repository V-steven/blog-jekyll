---
layout: post
title: 关于定时器setTimeout的解析
tags: [前端后台]
---

关于setTimeout的前端面试题，关于for循环、定时器setTimeout()、JavaScript闭包、匿名函数和Promise等知识点。

---

## 1. setTimeout知识点

**（1）setTimeout() 方法用于在指定的毫秒数后调用函数或计算表达式**

**（2）理解执行流程** 

执行引擎先将setTimeout()方法入栈被执行，执行时将延时方法交给内核相应模块处理。引擎继续处理后面代码，while语句将引擎阻塞了1秒，而在这过程中，内核timer模块在0.5秒时已将延时方法添加到任务队列，在引擎执行栈清空后，引擎将延时方法入栈并处理，最终输出的时间超过预期设置的时间。

```java
var start = new Date;
setTimeout(function(){
var end = new Date;
console.log('Time elapsed:', end - start, 'ms');
}, 500);
while (new Date - start < 1000) {};
//==>Time elapsed:1500ms
```

任务放在执行队列（先进先出）中，队列执行的时间，需要等待到函数调用栈清空之后才开始执行。即所有可执行代码执行完毕之后，才会开始执行由setTimeout定义的操作。即使我们将延迟时间设置为0，它定义的操作仍然需要等待所有代码执行完毕之后才开始执行。如下：

```java
var timer = setTimeout(function() {
    console.log('1');
}, 0);

console.log('2');
//==>2,1
```

## 2. 解析

**（1）简单的for循环**

```java
for (var i = 0; i < 5; i++) {
  console.log(i);
}
//==>0,1,2,3,4
```

**（2）setTimeout会延迟执行，执行console时，i已经变成了5，最终输出5个5**

```java
for (var i = 0; i < 5; i++) {
  setTimeout(function() {
    console.log(i);
  }, 1000 * i);
}
//for循环，累加到5时不执行
//==>5,5,5,5,5
```

**（3）加个闭包就可以实现自己想要的结果**

```java
for (var i = 0; i < 5; i++) {
  (function(i) {
    setTimeout(function() {
      console.log(i);
    }, i * 1000);
  })(i);
}
//==>0,1,2,3,4
```

**（4）删掉其中的i，函数内部将不会对i保持引用**

```java
for (var i = 0; i < 5; i++) {
  (function() {
    setTimeout(function() {
      console.log(i);
    }, i * 1000);
  })(i);
}
//==>5,5,5,5,5
```

**（5）给setTimeout传递个立即执行函数，立刻执行函数**

```java
for (var i = 0; i < 5; i++) {
  setTimeout((function(i) {
    console.log(i);
  })(i), i * 1000);
}
//==>0,1,2,3,4
```

**（6）引发出来的promise知识点**

```java
setTimeout(function() {
  console.log(1)//定时，放入任务队列
}, 0);
new Promise(function executor(resolve) {
  console.log(2);//立即执行
  for( var i=0 ; i<10000 ; i++ ) {
    i == 9999 && resolve();
  }
  console.log(3);//立即执行
}).then(function() {
  console.log(4);
});//放到最后
console.log(5);//先5后4
//==>2,3,5,4,1
```