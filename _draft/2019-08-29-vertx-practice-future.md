---
title: Vert.x 实践 (2)：Future 介绍
tags: vertx 
---

之前我们在介绍异步回调的时候提到了回调嵌套的问题，也就是多层回调函数令代码呈现出一种很奇怪的风格，难以阅读。而为了解决这一问题，我们提到了 Future，在本文中将对它进行介绍。

对于没有接触过的同学来说，Future 的概念不是很显而易见，它的字面意思是“未来”，不像 String、Service、Node、Tree 等等这些直观的概念，很难想象一个叫未来的对象是什么东西。而且 Java、Scala 都实现有原生的 Future，它们和 vertx 有什么区别和联系，也是值得我们去了解的。

所以下面我们就从 Future 最基本的用法开始，逐步搞清楚这一概念究竟解决了什么问题。首先来看 Java 里面的 Future，我们知道，为了提高程序的响应时间，很多耗时操作可以放到子线程中去计算，主线程只需要负责接收请求、分发计算任务、返回结果。于是这就涉及到子线程怎么把计算结果返回给主线程的问题，Java 中的 Future 给出的解决方案是用 Future 对象来封装未来的计算结果，它暴露了几个方法

```java
// 查询任务是否已经执行完成
public boolean isDone();

// 查询任务是否取消 
public boolean isCancelled();

// 获得计算结果
public T get();
```

通过这些方法，我们能查询子线程任务的执行状态，从而进行一些必要的操作。这时 Future 的概念就很简单，相当于是未来产生对象的一个代理，能够回答当前询问者的一些关于任务状态的问题。Future 对象可以作为线程池任务提交的返回值

```java
ExecutorService service = Executors.newCachedThreadPool();
Future<Integer> f = executorServices.submit(() -> {
    // 模拟耗时操作
    Thread.sleep(1000);
    return 0;
});

// otherthings()

System.out.println(f.get());

```

需要注意的是，Future 对象的 get() 方法会阻塞当前线程，直到子线程任务执行完成才返回结果，所以上面的典型用法只能提供一定的并发能力。可以看到，java Future 的概念相当简单，能够很方便的从子线程中拉取计算结果。

接下来考虑 scala.concurrent 中提供的 Future，它同样可以作为向线程池提交任务的返回结果，不过由于 scala 的语法特性，Future 对象的创建方式和 java 版的不一样

```scala
def apply[T](body: =>T)(executor: ExecutionContext): Future[T]
```

Future 的 apply 方法是偏应用函数，第一个参数是具体的执行任务代码，第二个参数是 ExecutionContext 对象，也就是线程池对象。也就是说和 java 版相比其调用者反过来了，不过这并不重要，重要的是 scala Future 对结果的处理方式相比 java 版更进一步。在 java 中，是显式的把结果拉回来，这需要预判任务是否完成，如果没有则阻塞。而在 scala 中， 不需要把结果拉回到当前上下文，而是给 Future 对象注册回调函数，当任务完成后由 Future 对象自动调用。其实这就相当于把监控任务是否完成的职责交给了 Future 对象本身，而我们只需要传递给它完成之后要执行的动作就可以了。

```scala
// 导入 ExectionContext 对象到当前上下文，这将被隐式地赋予 Future.apply 方法的第二个参数
import scala.concurrent.ExecutionContext.Implicits.global

val future = Future[Int] {
    Thread(1000)
    10
}

// 注册任务成功后的动作
future.onSuccess{
      case i => println(i)
    }

// 注册任务失败后的动作
future.onFailure{
      case e: Exception => e.printStackTrace()
    }
    
//后续代码

```

所以在 scala 中，我们不必为任务何时完成而操心，相对于 java 原生的 Future 是一个进步。但事实上 java 中是存在可以注册回调函数的 Future 的，只不过它有另外一个名字叫 CompletableFuture，是 Future 的一个子类，用法都大同小异，这里就不细讲了。

vertx 中的 Future 概念其实和前面介绍的 Future 没有本质区别，不过在讨论它之前，我还想引入一个新的概念：Promise，这又是一个抽象的词，字面意思是“承诺”。下面我们来强行扯一扯它 Future 的关系：当你许下一个承诺时，被许诺人知道你肯定不能马上实现诺言，所以她不会傻傻地在原地等你，而是该干嘛干嘛。但是她心里会有一个预期，既然是承诺，就是在“未来”某个时刻，它要么成功实现，要么失败，所以 Promise 是自带有 Future 属性的。所以在程序中，promise 和 future 可以像下面这样来使用

```scala
p = new Promise() // 许下一个承诺
p.future.onSuccess(result -> ...)// 兑现承诺后应该做的事情
p.future.onFailure(e -> ...) // 发现无法兑现承诺时应该做的事情

// 接下来该干嘛干嘛

// 在异步环境中
result = workHard()
if(result == success) 
  p.success(result) // 兑现承诺
else
  p.failure() // 无法兑现承诺
```

个人认为按这种方式理解 Promise 比较直观点，学术上把 Promise 称为“一个可写的单一赋值容器”就有点不太好理解了。

现在我们来看 vertx 中的 Future 用法，与 Promise 搭配起来的样子如下

```java
Promise<Integer> promise = Promise.promise();
promise.future().setHandler(ar -> {
            if(ar.succeeded()) {
                System.out.println(ar.result());
            }else {
                ar.cause().printStackTrace();
            }
        });

new Thread(() -> {
            boolean success = doSomething();
            if(success) {
                promise.complete(1);
            }else {
                promise.fail("some reason");
            }
            
        }).start();

// doOtherThings()

```

这一过程和上面的伪代码几乎一样，只不过在 vertx 中设置回调函数用的是 setHandler 方法。
