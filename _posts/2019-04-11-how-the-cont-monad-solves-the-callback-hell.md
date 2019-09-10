---
layout: post
title: Cont Monad 是如何拯救回调地狱的
tags: 函数式编程 Monad Java
---

### 回调方法

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
在 query 方法内部，临时创建线程还是使用线程池，都是库开发者可以灵活选择的。

### 多层回调

以上，我们介绍了很简单的异步回调方法，但是只涉及到了一层回调，在异步编程的环境中，几乎任何耗时操作都应该异步执行，也就是说需要额外传一个回调方法作为参数。再举一个例子

```java
double a = 1.0;
double b = a + 2.0;
double c = b + 3.0;
double d = c + 4.0;
System.out.println(d);
```

这是我们的原型，下面把它转换成回调的版本，首先对 double 类型进行封装

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

### Cont Monad 简介

Monad 本身是一个不太好理解的数学概念，但是我们在这里还是以实用为主，不会涉及得很深，所以通常讲 Monad 需要提到的 haskell 代码我们也不写了，而是直接用 Java 来描述。如果我们将 Monad 看作是 Java 中的一个类的话，那么这个类主要有两个行为，第一是生成 Monad 对象，这里我们用 pure 函数表示，第二是绑定一个操作，用 bind 函数表示，下面给出了一个很简单的实现

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

但是上面这个简单的 Monad 并不足以解决我们的问题，我们真正需要的是 Cont Monad。为了搞清楚 Cont Monad 究竟是什么，我们先来了解 CPS 函数，CPS 全称 continuation-passing style，字面意思是连续传递风格的函数，这里的传递指的是参数的传递，什么意思呢？举个简单的例子，一个进行加法运算的普通函数定义如下

```java
public double plus10(double a) {
    return 10 + a;
}
```

如果我们要对返回值进行其他操作，比如打印出来，则需要把返回值传给打印函数

```java
double result = plus10(10);
System.out.println(result);
```

而对于 CPS 函数来说，它不返回结果，而是把结果继续传给另一个函数，而这“另一个函数”就是 continuation，下面我们定义 CPS 版本的 plus10

```java
public Void plus10CPS(double a, Function<Double, Void> f) {
    double result = 10 + a;
    return f.apply(result);
}
```

该函数除了接收本来的参数 a 以外，还接收一个函数 f，并在函数体中，将结果传给 f，很显然，continuation 其实就是一个回调函数。如果要打印结果可以像下面这样

```java
plus10CPS(10, res -> {
    System.out.println(res);
    return null;
})
```

若把 plus10 和 plus10CPS 都改成 lambda 表达式，能更好地对比两者的特点 

```java
Function<Double, Double> plus10 = a -> a + 10;
Function<Double, Function<Function<Double, Void>, Void>> plus10CPS = a -> f -> {
            return f.apply(a + 10);
        };

//apply
double d = 10.0;
double res = plus10.apply(d);
System.out.println(res);

plus10CPS.apply(d).apply(res -> System.out.println(res));
```

可以看到，plus10 与 plus10CPS 的区别在于返回值类型不同，前者返回 Double，后者返回 Function<Function<Double, Void>, Void>，而且 cps 函数的使用有一个特点，需要首先 apply 值，然后 apply 一个函数。下面我们定义第二个 cps 函数

```java
Function<Double, Double> mul2 = d -> d * 2;

Function<Double, Function<Function<Double, Void>, Void>> mul2CPS = a -> f -> {
            return f.apply(a * 2);
        };
```

然后考虑 plus10 和 mul2 的函数组合，即先加 10 再乘以 2，用普通函数表达如下

```java
Function<Double, Double> plus10ThenMul2 = a -> mul2.apply(
    plus10.apply(a)
)
```

而 CPS 版本的函数组合则稍微复杂点

```java
Function<Double, Function<Function<Double, Void>, Void>> plus10ThenMul2CPS = a -> f -> {
    return plus10CPS.apply(a).apply(res -> 
                        mul2CPS.apply(res).apply(f)
                    );
}
```

CPS 函数的使用方法类似于 cps.apply(a).apply(f)，所以上面的过程就是先把参数 a 传给 plus10CPS，然后把结果传给 mul2CPS，最后得到组合后的 CPS 函数 plus10ThenMul2CPS，之所以要费力地把函数组合写成 CPS 形式，是因为后面我们将会看到这与 Cont bind 的实现相关，要理解 bind 里面的操作，最好先理解这里 CPS 函数的组合。

<!-- 当接收一个 Double 类型的参数后，plus10CPS 返回了一个 Function<Function<Double, Void>, Void> 类型的值，这仍然是一个函数，它接收 continuation，并返回最终的处理结果，当然这里的最终返回类型是 Void，因为我们只是想打印结果值。但如果我们想返回数值，则可以将 Void 替换成相应类型，比如

```java
Function<Double, Function<Function<Double, Double>, Double>> plus10CPS = a -> f -> {
    return f.apply(a + 10);
};

double result = plus10CPS.apply(5).apply(res -> {
    return res * 2;
});
``` -->

假如我们把普通函数，比如 plus10 称作原始函数，那么它的 CPS 版本接收了原始函数的参数（在这里就是 a）之后，则会返回一个类型如 Function<Function<A, R>, R> 的函数，这里我们暂且叫它 c 函数，那么这个函数究竟是什么含义呢？我们可以从回调的角度来理解，前面提到 continuation 其实就是一个回调函数，当 c 函数接收 continuation（即回调函数）（在例子中为 f） 后，会调用 continuation 的 apply 方法应用到计算结果上（即 a + 10），这一过程其实就是“回调函数的执行”，也就是说 c 函数内部封装了回调函数的执行动作，所以下面我们将 c 函数称为 execCallback。

从另一方面来看，我们在执行函数的时候需要两个材料，首先是函数本身，然后才是参数

```java
public Void f(double a) {
    System.out.println(a);
    return null
}

......
// later
double a = 10.0;
f(a);

```

在通常情况下，函数 f 就定义在那里，是我们知道的，参数是需要后续传递的，在正式计算之前是未知的。但如果把这两个材料看作是另一个高阶函数的两个参数的话，则可以定义出下面的函数

```java
public R apply(Function<A, R> f, A a) {
    return f.apply(a);
}
```

这种形式方便我们看到另一种情况，即参数先于函数被知晓，而 execCallback 便提供了这种抽象，举个例子来说

```java
double a = 10.0;
Function<Function<Double, Void>, Void> execCallback = f -> f.apply(a);

....
// later
execCallback.apply(res -> {
    System.out.println(res);
    return null;
});
```

<!-- 当存在多个 CPS 函数时，我们可以把它们组合起来完成一系列运算

```java
// plus 10 cps version
Function<Double, Function<Function<Double, Double>, Double>> plus10CPS = a -> f -> {
    return f.apply(a + 10);
};

// mul 2 cps version
Function<Double, Function<Function<Double, Double>, Double>> mul2CPS = a -> f -> {
    return f.apply(a * 2);
};


``` -->



下面我们来到对 Cont 的讨论，其实 Cont 正是对 execCallback 的封装，封装方法为

```java
public static Cont<R, A> cont(Function<Function<A, R>, R> execCallback){
    return new Cont<>(execCallback);
}
```

相应的，为了向 execCallback 传入回调函数，Cont 需要一个入口

```java
public R runCont(Function<A, R> f) {
    return execCallback.apply(f);
}
```

前面也提到，execCallback 抽象了参数先于函数的情况，那么通过值（而非函数）也可以构造 Cont，即下面的 pure 

```java
public static <R, A> Cont<R, A> pure(A a) {
    return cont(f -> f.apply(a));
}
```

由于 execCallback 是 CPS 函数的返回值，所以 CPS 函数又可以写成 Cont 版本的，
下面把 plus10CPS 和 mul2CPS 的返回值用 Cont 进行封装

```java
Function<Double, Cont<Void, Double>> plus10Cont = a ->
                Cont.cont(f -> f.apply(a + 10));

Function<Double, Cont<Void, Double>> mul2Cont = a ->
                Cont.cont(f -> f.apply(a * 2));
}
```

前面我们提到 CPS 函数的组合，自然其 Cont 版本也存在这种组合

```java
Function<Double, Cont<Void, Double>> plus10ThenMul2Cont = a -> Cont.cont(
    f -> {
        return plus10Cont.apply(a).runCont(res -> {
            return mul2Cont.apply(res).runCont(f);
        });
    }
);
```

更有甚者，还可以使用 CPS 版本函数和 Cont 版本函数的混合组合

```java
Function<Double, Cont<Void, Double> plus10ThenMul2Cont = a -> Cont.cont(
    f -> {
        return plus10CPS.apply(a).apply(res -> {
            return mul2Cont.apply(res).runCont(f);
        })
    }
);
```

这里 plus10CPS.apply(a) 的返回结果是一个 execCallback，由于 Cont 封装了一个 execCallback，所以其 bind 方法省去了第一步 apply，实现如下

```java
public <B> Cont<R, B> bind(Function<A, Cont<R, B>> k) {
    return cont(f -> {
        return execCallback.apply(res -> {
            return k.apply(res).runCont(f);
        })
    })
}
```

也就是说，bind 方法组合的两个函数，第一个函数由 Cont 本身封装，第二个函数作为参数传入，当然 execCallback 的 apply 方法可以通过 runCont 方法间接执行

```java
public <B> Cont<R, B> bind(Function<A, Cont<R, B>> k) {
    return cont(f -> {
        return runCont(res -> {
            return k.apply(res).runCont(f);
        })
    })
}
```

经过以上讨论，我们给出 Cont 的完整形式


<!-- 由于 Cont 是 Monad，所以它还有 bind 方法，其实现如下

```java
public <B> Cont<R, B> bind(Function<A, Cont<R, B>> k) {
        Function<Function<B, R>, R> f = callback -> runCont(a -> k.apply(a).runCont(callback));
        return cont(f);
}
``` -->

<!-- 这里的 Cont 代表 continuation。continuation 其实是一个术语，可以理解成“程序剩下的部分”（By 知乎@圆桌骑士魔理沙），但还是不太明了，我们来细致分析一下，既然提到了“程序剩下的部分”，那么肯定还隐含两个概念，那就是“完整的程序”和“程序已有的部分”。举一个简单的例子，如果要拼接字符串 "hello" + " Cont"

```java
"hello" + " Cont";
```

对于 "hello" 来说，它自身是已有的部分，则剩下的部分就为

```java
Function<String, String> f = str -> str + " Cont";
```

于是完整的程序就可以重写为

```java
//已有的部分
String str = "hello";

//剩余的部分
Function<String, String> f = str -> str + " Cont";
f.apply(str);
```

上面例子中程序的目的是很明确的，但如果我们还不知道完整的程序是什么，那么该如何表示程序剩下的部分呢？比如已知字符串 "hello "，只知道程序会用到它，但并不知道会怎么用它，我们仍借助占位符

```java
Function<Function<String, B>, B> left = f -> f.apply("hello");
```

其中 f 是接收 String 类型的函数，其返回类型未知，我们用泛型来表示。left 就是我们已知 "hello" 后程序剩下的部分，也就是 continuation。当我们在之后的某个时间决定了该怎样使用已知的 "hello" 后，便可将此动作作为参数传给 left，比如打印操作

```java
left.apply(s -> {
    System.out.println(s);
    return null;
});
```

这里之所以要返回 null，是因为 left 接收的参数是一个 Function，必须有一个返回值，此时 B 的具体类型可以写成 Void。

从上面的描述中我们能发现 continuation 接收的参数其实其实是一个回调函数，之后我们将看到，正是因为这层关系，才使得 Cont Monad 能解决回调地狱的难题。接下来我们讨论 Cont 的定义，首先，Cont 是一个 Monad，拥有 pure 和 bind 方法，其次 Cont 封装了一个 continuation，方法为 cont，最后，为了将具体操作传给 continuation，Cont 还拥有 runCont 方法，具体的实现代码如下 -->

<!-- 
可以简单理解成回调，或者更准确的说是执行回调函数的动作，这里的差别很微妙，比如回调函数我们可以写成

```java
Function<A, B> callback = a -> func(b);
```

但是执行回调函数的动作就不一样了

```java
Function<<Function<A, B>, B> action = callback -> callback.apply(a);
```

回调函数的输入类型是 A，输出类型是 B，而执行回调函数的动作的输入是一个回调函数，输出是该回调函数的输出类型。Cont 就像前面的 Monad 一样，里面包裹了某种东西，不同的是，Cont 包裹的是执行回调函数的动作，另外还有两个函数 cont 和 runCont，cont 函数负责接收一个执行回调函数的动作，并将其封装后返回 Cont 对象，而 runCont 则接收一个具体的回调函数，并执行该 Cont 所封装的动作。具体代码如下 -->

```java
public class Cont<R, A> {

    private Function<Function<A, R>, R> execCallback;

    public Cont(Function<Function<A, R>, R> execCallback) {
        this.execCallback = execCallback;
    }

    public static <R, A> Cont<R, A> pure(A a) {
        return cont(callback -> callback.apply(a));
    }

    public static <R, A> Cont<R, A> cont(Function<Function<A, R>, R> execCallback) {
        return new Cont<>(execCallback);
    }

    public R runCont(Function<A, R> fn) {
        return execCallback.apply(fn);
    }

    public <B> Cont<R, B> bind(Function<A, Cont<R, B>> k) {
        return cont(f -> runCont(a -> k.apply(a).runCont(f)));
    }
}

```

现在，利用 Cont 可以将我们的求和过程写成连贯的形式

```java
Cont.<Void, DWrapper>pure(new DWrapper(1))
    .<DWrapper>bind(w -> Cont.cont(cb -> w.plus(2.0, cb)))
    .<DWrapper>bind(w -> Cont.cont(cb -> w.plus(3.0, cb)))
    .<DWrapper>bind(w -> Cont.cont(cb -> w.plus(4.0, cb)))
    .runCont(w -> {
        System.out.println(w);
        return null;
    });

System.out.println("end");
```

运行代码，可以看到先打印出了 "end"，说明前面的连续处理是异步的，并且没有回调地狱的问题。

### 总结

理解 Cont Monad，如果直接看源码的话，会显得相当困难，尤其是 bind 函数的实现，给人的感觉就是，虽然起作用，但是很懵。所以，我觉得先搞清楚 CPS 函数的组合，再到返回 Cont 版本的函数组合，能够从原理上讲清楚为什么是那样的。




<!-- 虽然 pure 函数和 cont 一样都返回 Cont 对象，但是 pure 接收的是具体类型的值，它起的作用是将这个值置于某个潜在回调函数的执行氛围里，然后在 runCont 方法中得到真正的执行。我们知道，在使用异步回调时，比如 DWrapper 的 plus 方法里面，需要把运行结果传给回调函数，这一过程在 Cont 中就被分为了两个函数，即 pure （封装结果）和 runCont（运行回调函数）。这两个函数使用起来就像下面这样

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

k 在应用 a 之后表达式变换为

```java
f = callback -> 
        runCont(a -> 
            Cont.cont(cb -> cb.apply(a + " Cont"))
                .runCont(callback)
        );
```

继续规约内部那个 runCont
```java
f = callback -> 
        runCont(a -> 
            (cb -> cb.apply(a + " Cont")).apply(callback)
        )

  = callback -> runCont(
        a -> callback.apply(a + " Cont")
    )
```

再继续执行 runCont 的话，其调用者就应该是前面提到的 

```java
applyCallback = callback -> callback.apply("hello")
```

为了避免混淆，我们将 bind 里面的 callback 修改成 callback2，于是有

```java
  f = callback2 -> 
        applyCallback
            .apply(a -> callback2.apply(a + " Cont"))

    = callback2 -> 
        (callback -> callback.apply("hello"))
            .apply(a -> callback2.apply(a + " Cont"))
        
    = callback2 -> (a -> callback2.apply(a + " Cont")).apply("hello")

    = callback2 -> callback2.apply("hello" + " Cont")
```

通过上述一系列规约，我们看到了 bind 函数执行了字符串拼接，并将其作为参数再次构造了一个“执行回调函数的动作”，使用 cont 封装成 Cont 返回。最后执行 

```java
 .runCont(message ->{
     System.out.println(message);
     return null;
 });
```

就可以表述为

```java
Cont.cont(callback2 -> callback2.apply("hello" + " Cont"))
    .runCont(message ->{
        System.out.println(message);
        return null;
    })

    = callback2 -> callback2.apply("hello Cont")
        .apply(message ->{
            System.out.println(message);
            return null;
        })
    
    = (message -> {
            System.out.println(message);
            return null;
        }).apply("hello Cont")

// output:
// hello Cont
```

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
        runCont(a -> k.apply(a).runCont(callback))
    // 替换 k = w -> Cont.cont(cb -> w.plus(2, cb))
    = callback -> runCont(a -> 
        (w -> Cont.cont(cb -> w.plus(2, cb)))
        .apply(a)
        .runCont(callback)
    )
    // 利用 a 规约 w
    = callback -> runCont(a -> 
        Cont.cont(cb -> a.plus(2, cb))
        .runCont(callback)
    )
    // 利用 Cont.cont(lambda).runCont(callback) = lambda.apply(callback)
    = callback -> runCont(a -> 
        (cb -> a.plus(2, cb)).apply(callback)
    )
    // 利用 callback 规约 cb，为了区分，使用 callback2 替换 callback
    = callback2 -> runCont(a -> 
        a.plus(2, callback2)
    )
    // 此 applyCallback 是本 Cont 封装的“执行回调函数的动作”
    = callback2 -> applyCallback.apply(a -> a.plus(2, callback2))
    
    = callback2 -> 
        (callback -> callback.apply(new DWrapper(1.0)))
                .apply(a -> a.plus(2, callback2))
        
    = callback2 -> 
        (a -> a.plus(2, callback2))
            .apply(new DWrapper(1.0))
    
    = callback2 -> new DWrapper(1.0).plus(2, callback2)

```

可以看到，bind 函数中的一系列转换最终生成了一个以回调函数为参数的 lambda 表达式，并执行了加法运算 -->