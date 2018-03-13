---
layout: post
title: JavaScript的this作用域解析
tags: [前端后台]
---

学习JS，对于其中的this作用域有很大的迷惑，在面向对象的其他语言中，指向当前对象。JS中，`this`到底指向谁，下面我们来分析分析....

---

# 1.全局函数的调用

**全局性的方法，那么为全局性的调用，则`this`则代表全局对象window.**

```java

alert(this);// this===window

//附录：==用于比较，判断两者是不是相等，若两者数据类型不同则自动转换；
//===用于严格比较，两者是否相等，同时要求数据类型也要一摸一样； 


//调用全局方法，即全局性的作用域
var name = "global this";

function globalTest() {
    console.log(this.name);
}
globalTest(); //global this

//创建新对象，则this指向的是新对象的作用域
var test = function(){
  alert(this === window);
 }
new test(); //false


```

**全局变量在方法中被修改**

```java
    var name = "global this";

    function globalTest() {
        this.name = "rename global this"
        console.log(this.name);
    }
    globalTest(); //rename global this
```
# 2.构造函数的调用

**构造函数中的`this`指向的对象本身**

```java
function showName() {
        this.name = "showName function";
    }
    var obj = new showName();
    console.log(obj.name); //showName function

```

# 3.apply/call的this调用

**应用某一对象的一个方法，用另一个对象替换当前对象**
```java
var value = "Global value";

function FunA() {
    this.value = "AAA";
}

function FunB() {
    console.log(this.value);
}
FunB(); //Global value 因为是在全局中调用的FunB(),this.value指向全局的value
FunB.call(window); //Global value,this指向window对象，因此this.value指向全局的value
FunB.call(new FunA()); //AAA, this指向参数new FunA()，即FunA对象

FunB.apply(window); //Global value
FunB.apply(new FunA()); //AAA

///////

var test = function(){
  alert(this === window);
 }
var test1 = {
}
test.apply(test1);//window对象被test1对象给替代了
```

