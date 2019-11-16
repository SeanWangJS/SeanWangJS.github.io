---
layout: post
title: Vert.x 实践 (1)：异步回调
tags: vertx
---

本系列文章主要是想记录下我在学习和使用 vertx 框架过程中总结的一些经验技巧，同时巩固对 vertx 的理解。

Vert.x 是 Java 中的异步编程框架，当然还支持除 Java 之外的几种语言，在我看来，它以很现代化的方式为 Java 编程带来了一种新的体验。说到异步编程，就不得不提回调函数这种很有用的概念，在 vertx 中，无处不在的异步调用让我们不得不最先对回调函数进行仔细介绍。

在同步编程的环境中，调用一个函数之后，程序必须等待函数返回再执行下一行代码，也就是说，程序的执行顺序和我们的编码顺序是相同的，所以同步编程的好处就是让程序员可以很容易掌握程序的执行过程和状态，方便调试。而它的缺点也很明显，那就是不太容易充分利用资源，因为在等待函数返回的时候，程序实际上什么都做不了，如果被等待的函数只耗时，不耗资源，那么这段时间 CPU 相当于处于闲置状态，所以很难提高同步程序的响应时间和吞吐量。

而异步编程则能有效解决在同步环境中遇到的这些问题，回调函数是为异步编程提供的一种解决方案，和同步编程不一样，它在调用函数的时候，不会等着返回结果，而是以发射后不管的方式继续执行后面的代码。这时候就遇到一个问题，我们调用函数的目的很大程度上就是想让它帮我们计算一些值，然后供下一步使用，现在不等待函数返回，后续的代码怎么运行呢？答案就是把依赖于函数调用结果的代码封装成回调函数，再传给原本要调用的函数，这么说有点拗口，下面的代码更直观一点

```scala

// 同步版本，1、传参调用函数 foo， 2、等待返回 3、将结果传给函数 bar
val result = foo(params...)
bar(result)

// 异步版本，1、将后续计算封装成回调函数 2、将参数和回调函数一并传给 foo
val callback = result => bar(result)
foo(params..., callback)
```

在 vertx 中，几乎所有的函数调用都是异步的，比如创建 Http 服务器、发送消息、查询数据库等等

```java
// 创建 Http 服务器
vertx.createHttpServer()
  .listen(8080, ar -> {
    if(ar.succeed()) {
      ...
    }else {
      ...
    }
  });

// 向某个地址发送消息
vertx.eventBus().send("address", message, ar -> {
  if(ar.succeed()) {
    ...
  }else {
    ...
  }
});

// 访问数据库
jdbcClient.querySingle("select * from table", ar -> {
  if(ar.succeed()) {
    ...
  }else {
    ...
  }
});
```

从上面几个例子可以发现，函数调用最后的参数都是一个 lambda 表达式，我们通过它定义函数调用完成之后的操作，其中的 ar 是 AsyncResult 的简写，即异步结果，通过 ar 我们可以访问函数执行完成后的状态和值，如果执行成功，则 ar 的状态为 succeed，否则为 failed。

另一方面可以看到，异步回调的编程方式迫使我们把一连串任务的代码组织到一起，从而达到一个函数完成一件事情的目的，所以实际上对我们的代码设计也是有好处的。但这也有一个弊端，如果连续嵌套多个回调函数，会让代码看起来相当难看，导致回调地狱的情况

```scala
// 同步版本，分别计算 res1, res2, res3, 再传入 bar
val res1 = foo1(p1)
val res2 = foo2(p2)
val res3 = foo3(p3)
bar(res1, res2, res3)

// 异步版本，回调嵌套导致代码可读性骤降
foo1(p1, res1 => {
  foo2(p2, res2 => {
    foo3(p3, res3 => {
      bar(res1, res2, res3)
    })
  })
})
```

为了解决这一问题，我们将在后面介绍 Future 的用法，本篇的目的主要是是熟悉 vertx 中的异步回调编程风格。为了加深理解，下面我们实现一个自己的异步回调函数，将 jdbc 查询过程封装成异步形式，首先我们给出同步版本的查询函数

```java
// 查询函数定义
public ResultSet query(String sql) {

  Statement stat = conn.createStatement();
  return stat.executeQuery(sql);

}

// 调用函数
String sql = "select username from table where age = 18";
ResultSet resultSet = query(sql);
List<String> names = new ArrayList<>();
while(resultSet.next()) {

  names.add(resultSet.getString(0));

}

System.out.println(names);

//后续代码
...
```
在上面的代码中，即便后续代码与前面的无关，也必须等到前面的执行完成之后才能开始运行。现在我们把调用函数的代码封装成回调函数，并传给查询函数

```java
// 回调版本的查询过程
public void query(String sql, Function<ResultSet, Void> callback) {

  Statement stat = conn.createStatement();
  callback.apply(stat.executeQuery(sql));

}

// 调用函数
String sql = "select username from table where age = 18";
query(sql, resultSet -> {
  List<String> names = new ArrayList<>();
  while(resultSet.next()) {

    names.add(resultSet.getString(0));

  }
  System.out.println(names);
  return null;
})

// 后续代码
```

可惜的是，上面这段代码只做到了形似异步调用，最关键的 query 函数内部过程仍然在当前线程中执行，并不能立即返回，所以这并不是真正的异步。修改方法也很简单，只需要把 query 函数内部的执行过程放到线程池中执行即可。

从以上的内容来看，异步回调其实是一个很简单的概念，关键是熟悉这种编程风格，我觉得这是学习 vertx 的首要要求。