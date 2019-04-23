---
title: Cont Monad 是如何拯救回调地狱的
---

回调是一种很有效的异步编程方法，举一个简单的例子，如果我们要执行一个数据库查询语句，通常会像下面这样

```java

Statement stat = ...;
String sql = ...;

ResultSet result = stat.executeQuery(sql);

//后续工作

```

其中，后续工作可能与查询结果有关，也可能无关。如果无关的话，那么等待查询结果返回的过程其实算是对时间的浪费，尤其是在对性能要求比较高的环境中，这种等待是不可接受的。解决的方法是把耗时操作放到另一个线程中执行

```java

new Thread(() -> {
    ResultSet result = stat.executeQuery(sql);
}).start();

//后续代码

```

但像上面这样显式的创建线程对客户端很不友好，一方面是线程创建的开销，另一方面也不利于代码的维护。一种比较好的方式是由库提供者负责线程的管理，向用户屏蔽具体的细节，也就是说，把多线程代码放到具体的执行方法内部，用户不可见。这样一来，执行方法便几乎是立即返回的（因为具体的执行过程放到了另一个线程中，当前线程不阻塞），但也就意味着，执行结果无法返回到客户端，这时候回调方法便派上用场了。回调方法相当于用户派出去追踪结果的制导武器，一旦执行结果出现，无论在哪里，回调方法便上前去消费掉。就像下面这样

```java

Function<ResultSet, Void> callback = res -> ...

client.query(sql, callback);

```

这里的 client 便是与数据库交互的客户端工具，可以用 jdbc 来实现

```java

public void query(String sql, Function<ResultSet, Void> callback) {
    new Thread(() -> {

        Statement stat = connection.createStatement();
        ResultSet res = stat.executeQuery(sql);
        callback.apply(res);

    }).start();
}

```
在 query 方法内部，最好的方式还是使用线程池，而这都是库开发者可以灵活选择的。

至此，我们介绍了很简单的异步回调方法，但是只涉及到了一层回调，在异步编程的环境中，几乎任何耗时操作都应该异步执行，也就是说需要额外传一个回调方法作为参数。我们再举一个例子

```java
double a = 1.0;
double b = a + 2.0;
double c = b + 3.0;
double d = c + 4.0;
System.out.println(d);
```

这是我们的原型，下面要把它转换成回调的版本，那么首先对 double 类型进行一下封装

```java
public class DWrapper {

    private double d;

    public DWrapper(double d) {
        this.d = d;
    }

    public Void plus(double v2, Function<DWrapper, Void> callback) {
        new Thread(() -> {
            try {
                // 模拟耗时操作
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            callback.apply(new DWrapper(d + v2));
        }).start();
        return null;
    }

    public double get() {
        return d;
    }

    @Override
    public String toString() {
        return d + "";
    }
}

```

在 plus 方法中，我们加了一句线程睡眠的代码是为了模拟耗时操作。接下来再把刚才的加法操作修改成回调版本

```java
new DWrapper(1).plus(2, res -> {
        res.plus(3, res2 -> {
            res2.plus(4, res3 -> {
                System.out.println(res3);
                return null;
            });
            return null
        });
        return null;
    });

System.out.println("end");

```

这里我们直接写了三层回调，由于 DWrapper 的 plus 方法是异步的，求和的代码瞬间返回，所以我们看到先打印出 "end"，然后才是 10。于是现在我们便看到了回调地狱是怎样产生的了，其实就是一系列连续的操作，每一步都依赖于前一步的结果，因为每步操作都是异步的，所以必须用回调的方式把它们串起来，否则前一步操作还没出结果，后一步操作就执行的话就错了。

可见，在异步编程的环境中，几乎“每行”耗时代码都是异步执行的，这里所说的“每行”指的是由 ";" 结尾的语句，比如前面回调版本的连续求和，因为它实际上的形式就像下面这样

```java
new DWrapper(1).plus(2, ...);
```

这种特点就导致了程序中凡是需要顺序执行的语句必须写成回调的形式，从而陷入回调的牢笼无法自拔。有一种简单的方式可以缓解这种现象，那就是把回调方法提前定义出来，就像下面这样

```java

Function<DWrapper, Void> c1 = res -> {
    System.out.println(res);
    return null;
};

Function<DWrapper, Void> c2 = res -> {
    res.plus(4, c1);
    return null;
}

Function<DWrapper, Void> c3 = res -> {
    res.plus(3, c2);
    return null;
}

new DWrapper(1).plus(2, c3);

```

可以看到，使用这种方式虽然避免了深度的回调，但是整个计算过程都反过来了，显得很别扭。接下来我们将介绍 Cont Monad 这种结构，来看看它是如何解决回调地狱问题的。

Monad 本身是一个很复杂的数学概念，但是我们在这里还是以实用为主，不会涉及得很深，所以通常讲 Monad 需要提到的 haskell 代码我们也不写了，而是直接用 Java 来描述。如果我们将 Monad 看作是 Java 中的一个类的话，那么这个类主要有两个行为，第一是生成 Monad 对象，这里我们用 pure 函数表示，第二是绑定一个操作，用 bind 函数表示，下面给出了一个很简单的实现

```java
public class Monad<A>{

    private A a;

    private Monad(A a) {
        this.a = a;
    }

    public static <A> Monad<A> pure(A a) {
        return new Monad<>(a);
    }

    public <B> Monad<B> bind(Function<A, Monad<B>> f) {
        return f.apply(a);
    }

}

```

使用上述 Monad，我们可以很方便地用流式api 表达顺序操作

```java
Monad.pure(1)
    .bind(a -> Monad.pure(a + 2))
    .bind(a -> Monad.pure(a + 3))
    .bind(a -> Monad.pure(a + 4));
```

当调用 Monad 的 bind 函数时，这里的参数 a 便被赋值为该 Monad 所包裹的值，换句话说就是 Monad 所包裹的值被绑定到 bind 的 lambda 表达式的参数上，这也是之所以这个函数叫 bind 的原因。

但是上面这个简单的 Monad 并不足以解决我们的问题，我们真正需要的是 Cont Monad，这里的 Cont 代表 continuation。continuation 其实是一个术语，可以简单理解成回调，或者更准确的说是执行回调函数的动作，这里的差别很微妙，比如回调函数我们可以写成

```java
Function<A, B> callback = a -> func(b);
```

但是执行回调函数的动作就不一样了

```java
Function<<Function<A, B>, B> action = callback -> callback.apply(a);
```

回调函数的输入类型是 A，输出类型是 B，而执行回调函数的动作的输入是一个回调函数，输出是该回调函数的输出类型。Cont 就像前面的 Monad 一样，里面包裹了某种东西，不同的是，Cont 包裹的是执行回调函数的动作，另外还有两个函数 cont 和 runCont，cont 函数负责接收一个执行回调函数的动作，并将其封装后返回 Cont 对象，而 runCont 则接收一个具体的回调函数，并执行该 Cont 所封装的动作。具体代码如下

```java
public class Cont<R, A> {

    private Function<Function<A, R>, R> applyCallback;

    public Cont(Function<Function<A, R>, R> applyCallback) {
        this.applyCallback = applyCallback;
    }

    public static <R, A> Cont<R, A> pure(A a) {
        return cont(callback -> callback.apply(a));
    }

    public static <R, A> Cont<R, A> cont(Function<Function<A, R>, R> applyCallback) {
        return new Cont<>(applyCallback);
    }

    public R runCont(Function<A, R> fn) {
        return applyCallback.apply(fn);
    }

    public <B> Cont<R, B> bind(Function<A, Cont<R, B>> k) {
        Function<Function<B, R>, R> f = callback -> runCont(a -> k.apply(a).runCont(callback));
        return cont(f);
    }
}

```

虽然 pure 函数和 cont 一样都返回 Cont 对象，但是 pure 接收的是具体类型的值，它起的作用是将这个值置于某个潜在回调函数的执行氛围里，然后在 runCont 方法中得到真正的执行。我们知道，在使用异步回调时，比如 DWrapper 的 plus 方法里面，需要把运行结果传给回调函数，这一过程在 Cont 中就被分为了两个函数，即 pure （封装结果）和 runCont（运行回调函数）。这两个函数使用起来就像下面这样

```java

Cont.<Void, String>pure("hello Cont")
    .runCont(message -> {
        System.out.println(message);
        return null;
    });

```

cont 函数起的作用很明了，它仅仅是接收“执行回调函数的动作”，然后将其封装并返回 Cont 对象。相比之下，bind 的实现就复杂多了，简直是蒂花之秀。为了理清它的逻辑，我们照着例子来看

```java
Cont.<Void, String>pure("hello")
    .<String>bind(s -> Cont.cont(cb -> cb.apply(s + " Cont")))
    .runCont(message -> {
        System.out.println(message);
        return null;
    });
```

这句代码做了三件事情，封装 "hello"；绑定字符串拼接动作 s + " Cont"；打印字符串。我们逐行分析，首先 pure 函数接收 "hello" 之后返回的 Cont 对象，其封装的内容为

```java
applyCallback = callback -> callback.apply("hello")
```

通过 bind 接收参数后，在 bind 函数里面

```java
k = s -> Cont.cont(cb -> cb.apply(s + " Cont"))
```

接下来我们重点看这一句

```java
Function<Function<B, R>, R> f = callback -> runCont(a -> k.apply(a).runCont(callback));
```

在我们的例子中，callback 的类型为 Function<String, Void>，a 的类型为 String，k 在应用 a 之后表达式变换为

```java
Function<Function<String, Void>, Void> f = callback -> runCont(a -> 
    Cont.cont(cb -> cb.apply(a + " Cont")).runCont(callback)
);
```

继续规约内部那个 runCont
```java
Function<Function<String, Void>, R> f = callback -> runCont(a -> 
    (cb -> cb.apply(a + " Cont")).apply(callback)

    = callback -> runCont(
        a -> callback.apply(a + " Cont")
    )
);
```

再继续执行 runCont 的话，其调用者就应该是前面提到的 

```java
applyCallback = callback -> callback.apply("hello")
```

为了避免混淆，我们将 bind 里面的 callback 修改成 callback2，于是有

```java
Function<Function<String, Void>, Void> f = callback2 -> 
    (callback -> callback.apply("hello"))
        .apply(a -> callback2.apply(a + " Cont"))
        
    = callback2 -> (a -> callback2.apply(a + " Cont")).apply("hello")

    = callback2 -> callback2.apply("hello" + " Cont")
```

通过上述一系列规约，我们看到了 bind 函数执行了字符串拼接，并将其作为参数再次构造了一个“执行回调函数的动作”，使用 cont 封装成 Cont 返回。

下面我们再分析一个例子巩固一下

```java
Cont.<DWrapper, Void>pure(new DWrapper(1.0))
    .<DWrapper>bind(w -> Cont.cont(callback -> w.plus(2, callback)))
    .runCont(w -> {
        System.out.println(w);
        return null;
    });
```

pure 函数接收 new DWrapper(1.0) 之后，返回的 Cont 对象内部封装的“执行回调函数的动作” 为

```java
applyCallback = callback -> callback.apply(new DWrapper(1.0))
```

然后通过 bind 方法接收参数后得到

```java
k = w -> Cont.cont(cb -> w.plus(2, cb));

Function<Function<DWrapper, Void>, Void> f = callback -> 
        runCont(a -> k.apply(a).runCont(callback)
    )

    = callback -> runCont(a -> 
        (w -> Cont.cont(cb -> w.plus(2, cb)))
        .apply(a)
        .runCont(callback)
    )

    = callback -> runCont(a -> 
        Cont.cont(cb -> a.plus(2, cb))
        .runCont(callback)
    )

    = callback -> runCont(a -> 
        (cb -> a.plus(2, cb)).apply(callback)
    )

    = callback -> runCont(
        callback -> a.plus(2, callback)
    )


```
