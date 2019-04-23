---
title: Java 中的 Monad（1）
---

对于没有接触过范畴论的人来讲，要从数学上理解 Monad 的概念着实不容易，而且，即便看懂了 Monad 的数学语言定义，把它对应到编程语言中的各种概念同样很困难。我觉得学习 Monad 比较好的方法是一开始接着地气，然后再一步步升华，所以我们首先来看看在 Java 中，Monad 是什么个样子。

下面是一段很简单，甚至有点无趣的代码

```java
int a = 1;
int b = a + 2;
int c = b + 3;
int d = c + 4;
System.out.println(d);
```

就是不断的对变量加上一个量然后再赋予新的变量，这段代码的特点是每一行代码都依赖于前一行，是一个顺序执行的关系。如果我们把它稍微封装一下，比如用 Content 类来封装值，然后再按上面的逻辑写一遍

```java
Content a = new Content(1);
Content b = a.plus(2);
Content c = b.plus(3);
Content d = c.plus(4);
System.out.println(d);
```

相较于第一段代码，封装后的加法显得繁琐不少，但是它也有个好处，那就是我们可以在 plus 方法里面搞事情，比如追加日志，于是 Content 类可以设计得像下面这样

```java
//code I
public class Content{
    private int value;
    private Logger logger = ...;
    public Content(int value) {
        this.value = value;
    }

    public Content plus(int v2) {
        logger.info("execute plus: " + value + " and "  + v2);
        int v = value + v2;
        logger.info("result: " + v);
        Content con =  new Content(v);
        return con;
    }
}
```

这就看到了面向对象的好处，我们可以封装通用的行为，从而减少重复代码。接下来我们把问题搞深刻一点，假设 plus 方法是一个很耗时的操作，并且在打印出 d 的后面还有代码等着执行，那么在单线程环境下后面的代码只能阻塞。我们再假设后续代码与 a b c d 没有任何关系，那么这种阻塞就没有意义了，于是我们可以考虑把 a b c d 的计算放到另一个线程中去

```java
// code II
new Thread(() -> {
    Content a = new Content(1);
    Content b = a.plus(2);
    Content c = b.plus(3);
    Content d = c.plus(4);
    System.out.println(d);
}).start();

//后续代码
System.out.println(1);
System.out.println(2);
```

这样一来，前后两段代码便可并发执行。但是现在有个问题，如果 Content 是我们开发的一个库中的类，那么用户必须显式的创建线程才能让代码异步执行，这就把线程管理的责任抛给了用户，显然这增加了用户的工作量。为了让我们自己拿回线程管理的权力，考虑将 plus 方法修改成回调版本

```java
//code III
public void plus(double v2, Consumer<Content> callback) {
        new Thread(() -> {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            callback.accept(new Content(v2 + value));
        }).start();
    }
```

在上面的代码中，让线程休眠 1000ms 是为了模拟耗时操作，plus 的第二个参数便为操作完成后的回调函数，在客户端它的用法如下

```java
// code IV
new Content(1)
    .plus(2, res -> {
        res.plus(3, res2 -> {
            res2.plus(4, res3 -> {
                System.out.println(res3);
            });
        });
    });

//后续代码
System.out.println(1);
System.out.println(2);

```

执行上述代码，控制台会先输出 1 2，最后才是 10。于是，我们以回调的方式实现了不让用户创建线程的目的，当然这里我们就遇到了著名的回调地狱现象。为了解决这一问题，我们可以尝试把 lambda 表达式显式的写出来

```java
// code V
Consumer<Content> c1 = con -> System.out.println(con);
Consumer<Content> c2 = con -> con.plus(4, c1);
Consumer<Content> c3 = con -> con.plus(3, c2);
new Content(1).plus(2, c3);
```

但是现在整个程序显得很别扭，代码的顺序和实际的运算过程是反着的。这是因为最后一行代码是立即执行的
```java
new Content(1).plus(2, c3);
```
在这里，如果 c3 为空，显然会抛出错误，所以必须先定义 c3，而 c3 又必须依赖于 c2，c2 依赖于 c1，所以导致整个过程反过来了。为了使顺序变得正常，我们必须在不具体定义 c3 的情况下，让编译器知道 c3 的存在，于是我们可以把计算封装在 lambda 表达式里面

```java
Content content = new Content(1);
Consumer<Consumer<Content>> cc1 = c3 -> content.plus(2, c3);
```

这样一来，plus 运算不是马上执行，而是在后面调用 cc1 的 accept 方法，并传入真正的 c3。下一步需要用到 c2，那么同样

```java
Consumer<Consumer<Content>> cc2 = c2 -> {
    // 利用 c2 构造 c3，同前面说的倒叙代码一样
    Consumer<Content> c3 = con -> con.plus(3, c2);
    cc1.accept(c3);
};
```

接下来用到 c1

```java
Consumer<Consumer<Content>> cc3 = c1 -> {
    // 利用 c1 构造 c2
    Consumer<Content> c2 = con -> con.plus(4, c1);
    cc2.accept(c2);
};
```

最后打印的时候，先构造 c1，然后将其传给 cc3

```java
Consumer<Content> c1 = con -> System.out.println(con);
cc3.accpet(c1);
```

把上面的一系列过程连起来

```java
// code VI
Content content = new Content(1);
Consumer<Consumer<Content>> cc1 = c3 -> content.plus(2, c3);

Consumer<Consumer<Content>> cc2 = c2 -> {
    Consumer<Content> c3 = con -> con.plus(3, c2);
    cc1.accept(c3);
};

Consumer<Consumer<Content>> cc3 = c1 -> {
    Consumer<Content> c2 = con -> con.plus(4, c1);
    cc2.accept(c2);
};

Consumer<Content> c1 = con -> System.out.println(con);
cc3.accept(c1);
```

我们先对上段代码的执行过程做一下分析，首先通过 new 关键字创建了 Content 的实例，其中的值等于 1，接着四个 lambda 表达式 cc1， cc2，cc3，c1，它们蕴含的代码都不会立即执行，最后向 cc3 传入 c1，于是就好似连锁反应一样，cc3，cc2，cc1 蕴含的代码依次执行，这个执行顺序刚好和代码块 code V 的执行顺序一样，但是定义的顺序却纠正过来了。

虽然有点繁琐，但好歹把顺序给调正了，接下来我们考虑把上述过程简化一下

```
---------------------
Content content = new Content(1);

Consumer<Consumer<Content>> cc1 = c3 -> content.plus(2, c3);

Consumer<Consumer<Content>> cc2 = c2 -> cc1.accept(content -> content.plus(3, c2));

Consumer<Consumer<Content>> cc3 = c1 -> cc2.accept(content -> content.plus(4, c1));

cc3.accept(content -> System.out.println(content));

---------------------
Content content = new Content(1);

Consumer<Consumer<Content>> cc1 = callback -> content.plus(2, callback);

Consumer<Consumer<Content>> cc2 = callback -> cc1.accept(content -> content.plus(3, callback));

Consumer<Consumer<Content>> cc3 = callback -> cc2.accept(content -> content.plus(4, callback));

cc3.accept(content -> System.out.println(content));

---------------------
Consumer<Consumer<Content>> cc1 = callback -> content.plus(2, callback);

Wrapper w1 = new Wrapper(cc1);

Consumer<Consumer<Content>> cc2 = callback -> cc1.accept(content -> content.plus(3, callback));
Wrapper w2 = new Wrapper(cc2);

Consumer<Consumer<Content>> cc3 = callback -> cc2.accept(content -> content.plus(4, callback));
Wrapper w3 = new Wrapper(cc3);
w3.exec(content -> System.out.println(content));

---------------------
Content content = new Content(1);
Consumer<Consumer<Content>> cc1 = callback -> content.plus(2, callback);
Wrapper w1 = new Wrapper(cc1);
w2 = w1.compose(cc1 -> callback -> cc1.accept(content -> content.plus(3, callback)))
w3 = w2.compose(cc2 -> callback -> cc2.accept(content -> content.plus(4, callback)))
w3.exec(content -> System.out.println(content));

---------------------

new Wrapper(cc1)
    .compose(cc1 -> callback -> cc1.accept(content -> content.plus(3, callback)))
    .compose(cc2 -> callback -> cc2.accept(content -> content.plus(4, callback)))
    .exec(content -> System.out.println(content));


new Wrapper(new Content(1))
    .compose(content -> c3 -> content.plus(2, c3))
    .compose(content -> c2 -> content.plus(3, c2))
    .compose(content -> System.out.println(content))

Content content = new Content(1);

new Wrapper(c3 -> content.plus(2, c3))
    .compose(c2 -> content -> content.plus(3, c2))
    .compose(c1 -> content -> content.plus(4, c1))
    .exec(content -> System.out.println(content));
    


(c1 -> 
    (c2 -> 
        (c3 -> content.plus(2, c3))
            .accept(content -> content.plus(3, c2)))
        .accept(content -> content.plus(4, c1)))
    .accept(content -> System.out.println(content))

compose:: Consumer<Content> -> Content -> Consumer<Content> -> Consumer<Content>

compose:: Content -> Wrapper<Content> -> Wrapper<Content> -> Wrapper<Content>

compose:: a -> m b -> m a -> m b

new Wrapper(new Content(1))
    .compose(con -> {
        return c -> con.plus(2, c);
    })

new Wrapper(new Content(1))
    .compose(con -> {
        return new Wrapper(callback -> con.plus(2, callback));
    })
    .exec(con -> {
        System.out.println(con);
    })


new Wrapper(new Result(1))
    .compose(res -> {
        return new Wrapper(res.plus(2));
    })
    .compose(res -> {
        return new Wrapper(res.plus(3));
    })
    .compose(res -> {
        return new Wrapper(res.plus(4))
    })
    .get();

public Wrapper<Result> compose(Function<Result, Wrapper<Result>> f);

Result -> Wrapper<Result>


new Content(1)
    .plus(2, res -> {
        res.plus(3, res2 -> {
            res2.plus(4, res3 -> {
                System.out.println(res3);
            })
        })
    })

a -> b -> f a -> f b

a -> f b -> f a -> f (f b)

a -> m b -> m a -> m b

ResultSet -> Future[ResultSet] -> Future[ResultSet] -> Future[ResultSet]


new Wrapper(callback -> new Box(1).plus(2, callback))
    .compose(cc1 -> callback -> cc1.accept(d -> new Box(d).plus(1, callback)))

compose:: Double -> Wrapper<Double> -> Wrapper<Double> -> Wrapper<Double>

Wrapper<Double>
    .compose(Double -> Wrapper<Double>)
    .compose(Double -> Wrapper<Double>)
    .exec(Double -> )


Consumer<Double> c1 = d -> System.out.println(d);
new Wrapper(c1)
    .compose(c1 -> d -> new Box(d).plus(4, c1))
    .compose(c2 -> d -> new Box(d).plus(3, c2))
    .compose(c3 -> d -> new Box(d).plus(2, c3))
    .exec(new Box(1));

Wrapper.wrap(1)
    .compose(d -> c3 -> new Box(d).plus(2, c3))



f1(a -> {
    f2(b -> {
        f3(b, id)
    })
})

callback:: Double -> IO ()
runCont:: (Double -> ()) -> ()
runCont = callback -> new Box(1).plus(2, callback)

callback:: Double -> Int
runCont:: (Double -> Int) -> Int
runCont = callback -> callback.apply(1.0)
```