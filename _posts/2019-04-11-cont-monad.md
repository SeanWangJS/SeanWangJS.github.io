---
layout: post
title: 利用 Cont Monad 封装计算
tags: 函数式编程 Monad Java
---

#### 从回调方法说起

回调是一种很有效的异步编程方法，举一个简单的例子，如果我们要执行一个数据库查询语句，通常会像下面这样

```java

Statement stat = ...;
String sql = ...;

ResultSet result = stat.executeQuery(sql);

//后续工作

```

其中，后续工作可能与查询结果有关，也可能无关。如果无关的话，那么等待查询结果返回其实是不必要的，如果对性能要求比较高，那这很显然会成为瓶颈。一个比较粗糙的解决方法是把耗时操作放到另一个线程中执行

```java

new Thread(() -> {
    ResultSet result = stat.executeQuery(sql);
}).start();

//后续代码

```

但像上面这样显式的创建线程对客户端很不友好，一方面是线程创建的开销，另一方面也不利于代码的维护。一种比较好的方式是由库提供者负责线程的管理，向用户屏蔽具体的细节，也就是说，把多线程代码放到具体的执行方法内部，用户不可见。这样一来，执行方法便几乎是立即返回的（因为具体的执行过程放到了另一个线程中，当前线程不阻塞），但也就意味着客户端无法感知执行结果。这时候回调方法便派上用场了，回调方法相当于用户派出去追踪结果的制导武器，一旦执行结果出现，无论在哪里，回调方法便上前去消费掉。就像下面这样

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

#### 连续传递风格 (Continuation Passing Style)

上面我们提到的传递回调函数的方法，在函数式编程领域中有一个与之对应但更深刻的概念叫 CPS，其中 continuation 就可以被看作是回调函数，它接受当前计算的结果作为参数，并在未来某个时刻运行。为了加深印象，我们做几组对比（为了配合函数式的风格，这里我们用 lambda 表达式来表示函数，也为习惯后续的 haskell 代码做准备）

```scala
val plus10: Int => Int = a => a + 10
val result = plus10(10)
println(result)
```

<!-- ```java
Function<Integer, Integer> plus10 = a -> a + 10
int result = plus10.apply(10);
System.out.println(result);
``` -->
这是普通函数，它的作用是对输入值加 10 后返回，然后打印出结果，很简单。接下来我们把它改造成 CPS 函数


```scala
val plus10CPS: Int => (Int => Unit) => Unit = a => f => f(a + 10)
val f: Int => Unit = res => println(res)
plus10CPS(10)(f)
```

<!-- ```java
Function<Integer, Function<Function<Integer, Void>, Void>> plus10CPS = a -> f -> f(a + 10);
Function<Integer, Void> f = res -> {
    System.out.println(res);
    return null;
};
plus10CPS.apply(10).apply(f);
``` -->

在上面的代码中 f 就是回调函数（只不过写成了 lambda 表达式形式），也就是 continuation。plus10CPS 在 apply 10 之后得到一个类型为 (Int => Unit) => Unit 的值，这是一个相当重要的概念，我们后面还会提到，在这里可以看到它接收 continuation 作为输入，并在内部把当前计算的结果（即 a + 10）传给 continuation。下面我们来看第二个例子

```scala
val mul2: Int => Int = a => a * 2
val result = mul2(10)
println(result)

val mul2CPS: Int => (Int => Unit) => Unit = a => f => f(a * 2)
val f: Int => Unit = res => println(res)
mul2CPS(10)(f)
```

<!-- ```java
Function<Integer, Integer> mul2 = a -> a * 2
int result = mul2.apply(10);
System.out.println(result);

Function<Integer, Function<Function<Integer, Void>, Void>> mul2CPS = a -> f -> f(a * 2);
Function<Integer, Void> f = res -> {
    System.out.println(res);
    return null;
};
mul2CPS.apply(10).apply(f);
``` -->

这个例子和前面没什么不同，只不过是把加 10 改成了乘 2，下面就是重点了，我们要把 plus10 和 mul2 这两个函数组合成一个更大的函数

```scala
val plus10ThenMul2: Int => Int = a => mul2(plus10(a))
```

<!-- ```java
plus10ThenMul2 = a -> mul2(plus10(a))
``` -->
普通函数的组合一目了然，但是 CPS 函数的组合则有点复杂，我们一步一步来，首先我们可以想到组合之后的函数的类型仍然是

```scala
val plus10ThenMul2CPS: Int => (Int => Unit) => Unit = a => f => {
    ...
}
```

<!-- ```java
Function<Integer, Function<Function<Integer, Void>, Void>> plus10ThenMul2CPS = a -> f -> {
    ....
}
``` -->
它第一个输入（即 a）是传给 plus10CPS 的

```scala
a => f => {
    val c = plus10CPS(a)
    ...
}
```

<!-- ```java
a -> f -> {
    c = plus10CPS.apply(a);
    ...
}

``` -->
这里的 c 需要接收一个 continuation ，但不是 f，因为 f 是 plus10ThenMul2CPS 的 continuation，c 需要接收的 continuation 肯定是 plus10CPS 之后的操作，不就是 mul2CPS 吗？但也不能把 mul2CPS 传给 c，因为类型不对，我们需要构造一个中间量，假设它是 f2

```scala
a => f => {
   val c = plus10CPS(a)
   val f2: Int => Unit = b => ...
   c(f2)
}
```

<!-- ```java
a -> f -> {
    c = plus10CPS.apply(a);
    Function<Function<Integer, Void> f2 = b -> ...
    return c.apply(f2);
}
 -->
<!-- ``` -->

前面我们提到，c 会把当前计算的结果（即 (10 + a)）作为参数传递给 continuation，于是 f2 的输入就是 (10 + a)，也就是说把 (10 + a) 的值绑定到了 b 上。下一步自然是把 b 传给 mul2CPS 

```scala
a => f => {
   val c = plus10CPS(a)
   val f2: Int => Unit = b => {
     c2 = mul2CPS(b)
     ...
   }
   c(f2)
}
```
<!-- ```java
a -> f -> {
    c = plus10CPS.apply(a);
    Function<Function<Integer, Void> f2 = b -> {
        c2 = mul2CPS.apply(b);
        ...
    }
    return c.apply(f2);
}

``` -->
mul2CPS在接收了 b 之后返回的 c2 仍然需要接收一个 continuation，并把当前计算结果（此时是 b * 2 ）传给它，因为函数组合操作已经完成了，所以这时的 continuation 就是 f。

```scala
a => f => {
   val c = plus10CPS(a)
   val f2: Int => Unit = b => {
     c2 = mul2CPS(b)
     c2(f)
   }
   c(f2)
}
```

<!-- ```java
a -> f -> {
    c = plus10CPS.apply(a);
    Function<Function<Integer, Void> f2 = b -> {
        c2 = mul2CPS.apply(b);
        return c2.apply(f);
    }
    return c.apply(f2);
}

``` -->

简化一下
```scala
val plus10CPSThenMul2CPS: Int => (Int => Unit) => Unit = a => f => plus10CPS(a)(b => mul2CPS(b)(f))
```
<!-- ```java
plus10CPSThenMul2CPS = a -> f -> plus10CPS.apply(a).apply(b -> mul2CPS.apply(b).apply(f))
``` -->


#### 回调嵌套问题

举一个简单例子

```java
double a = 1.0;
double b = a + 2.0;
double c = b + 3.0;
double d = c + 4.0;
System.out.println(d);
```

这是原型，它是一个顺序操作，并且除了第一行之外的每一行都依赖于前一行的结果，如果把它转换成回调的版本

```java
double a = 1.0;
add2(a, res1 -> {
    double b = a + 2;
    add3(b, res2 -> {
        double c = b + 3;
        add4(c, res3 -> {
            double d = c + 4;
            System.out.println(d);
        })
    })
});
```
这样的代码看起来就不那么舒服了，为了化解这种

<!-- 
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
-->

#### 使用 Cont 封装计算

在介绍 CPS 的时候，我们提到了一个重要的类型 (Int => Unit) => Unit ，或者更一般地 (A => R) => R，这是 一个 CPS 函数接收第一个参数后的返回类型，在 haskell 中，这个类型的名字就叫 Cont，也就是下面的 c2

```scala
val plus10CPS: Int => (Int => Unit) => Unit = a => f => f(a + 10)
val x: Int = ...
val c2: (Int => Unit) => Unit = plus10CPS(x)
========== //另一种表达方式
val c2: (Int => Unit) => Unit = f => f(x + 10)
```
从上面 c2 的第二种表达方式可以看到，Cont 是把当前计算结果 (x + 10) 传递给 continuation (即 f) 的抽象模型。为什么说它是抽象模型呢，因为 plus10CPS(x) 只完成了当前计算 (x + 10)，它的 continuation 在目前还不确定，而 Cont 模型表达了未来将要接收并应用 continuation 这一个行为。

<!-- Monad 本身是一个不太好理解的数学概念，但是我们在这里还是以实用为主。如果我们将 Monad 看作是 Java 中的一个类的话，那么这个类主要有两个行为，第一是生成 Monad 对象，这里我们用 pure 函数表示，第二是绑定一个操作，用 bind 函数表示，下面给出了一个很简单的实现

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

但是上面这个简单的 Monad 并不足以解决我们的问题，我们真正需要的是 Cont Monad。为了搞清楚 Cont Monad 究竟是什么，我们先来了解 CPS 函数，CPS 全称 continuation-passing style，字面意思是连续传递风格的函数，这里的传递指的是参数的传递，什么意思呢？举个简单的例子，一个进行加法运算的普通函数定义如下（为了让代码更简洁，下面我们用scala来写示例代码）

```java
val plus10: Int => Int = a => a + 10
```

如果我们要对返回值进行其他操作，比如打印出来，则需要把返回值传给打印函数

```scala
val result = plus10.apply(10)
println(result)
```

而对于 CPS 版本的函数来说，它除了接收一个参数作为输入之外，还接收另一个函数类型的参数来消费当前计算产生的结果

```scala
val plus10CPS: Int => (Int => Unit) => Unit = a => f => {
    val current = a + 10 // 当前计算
    f(current) // 消费当前计算的结果
}
```

该函数除了接收本来的参数 a 以外，还接收一个函数 f，并在函数体中，将结果传给 f，这里的 f 就是一个 continuation，很显然，continuation 其实和回调函数没啥区别，只不过在不同的理论体系下有不同的叫法而已。如果要打印结果可以像下面这样

```scala
plus10CPS(10)(r -> println(r))
```

下面我们定义第二个函数

```scala
// 普通版本
val mul2: Int => Int = d => d * 2 
// CPS 版本
val mul2CPS: Int => (Int => Void) => Void = a => f => f(a * 2)
```

接下来考虑 plus10 和 mul2 的函数组合，即先加 10 再乘以 2，用普通函数表达如下

```java
val plus10ThenMul2: Int => Int = a => mul2(plus10(a))
```

而 CPS 版本的函数组合则稍微复杂点

```java
val plus10ThenMul2CPS: Int => (Int => Unit) => Unit = a => f =>
      plus10CPS(a)(
          b => mul2CPS(b)(f)
          )
}
```

这里我们稍微解释一下，参数 a 和 f 跟前面定义的 CPS 函数中的意义一样。在函数内部，首先对 a 应用加 10 的操作——plus10CPS(a) ，没有问题，紧接着需要传入 continuation ，但是现在不能直接传入 f，因为 f 是“加10 乘 2”这一组合动作完成之后的后续操作。所以紧接着的 continuation 是把加 10 后的结果（也就是 b）传入 mul2CPS，最后再传入 f，完成函数组合。

之所以要费力地把函数组合写成 CPS 形式，是因为后面我们将会看到这与 Cont bind 的实现相关，要理解 bind 里面的操作，最好先理解这里 CPS 函数的组合。

<!--假如我们把普通函数，比如 plus10 称作原始函数，那么它的 CPS 版本接收了原始函数的参数（在这里就是 a）之后，则会返回一个类型如 (A => R) => R 的函数，这里我们暂且叫它 c 函数，那么这个函数究竟是什么含义呢？我们可以从回调的角度来理解，前面提到 continuation 其实就是一个回调函数，当 c 函数接收 continuation（即回调函数）（在例子中为 f） 后，会调用 continuation 的 apply 方法应用到计算结果上（即 a + 10），这一过程其实就是“回调函数的执行”，也就是说 c 函数内部封装了回调函数的执行动作，所以下面我们将 c 函数称为 execCallback。

如果我们拔高一下视角来观察函数调用这个行为，可以发现函数调用需要两个材料，首先是函数本身，然后才是参数

```java
// 材料1：定义一个函数
public Void f(double a) {
    System.out.println(a);
    return null;
}

// 材料2：赋予一个值
double a = 10.0;

// 调用函数
f(a);

```

如果把这两个材料看作是另一个高阶函数的两个参数的话，则这个高阶函数就是我们前面定义的 CPS 函数

```scala
val func: A => (A => R) => R = a => f => f(a)
```

当我们对它应用一个类型为 A 的值时，得到一个类型为 (A => R) => R 的东西，它的值为

```scala
f => f(a)
```

这个 lambda 表达式看起来很不寻常，因为在一般的观念中，下面这种才是正经的 lambda 表达式

```haskell
\x -> f(x) 
``` -->

在 haskell 中 Cont 的声明如下

```haskell
newtype Cont r a = Cont {runCont:: (a -> r) -> r}
```
这一句为 (a -> r) -> r 这个类型赋予了一个名字叫 Cont r a，现在假如有一个值 c，给出两种等价的表达

```haskell
c1:: Cont r a
c2:: (a -> r) -> r
```
虽然两者定义上等价，但实际上，c2 可以接收参数，并且这个参数就是我们前面提到的 continuation，而 c1 则不行。为了让 c1 也能计算，需要使用 runCont 把真正的类型 (a -> r) -> r 暴露出来，runCont 的类型正是

```haskell
runCont:: Cont r a -> (a -> r) -> r
```
所以，Cont 更像是对 (a -> r) -> r 类型的一种封装，基于这样的认识，我们可以在 scala 中这样定义

```scala
class Cont[R, A](c2: (A => R) => R) {
  
  def runCont(f: A => R) = c2(f) 

}
```
此时，runCont 的参数是 continuation。假如有一个 Cont 类型的实例 c1，那么 

```scala
c1.runCont
```

才是和 c2 等价的，因为它们都接收一个 continuation 作为参数。

<!-- 那么在 haskell 中的

```haskell
runCont cont f
```
等价于 scala 中的
```scala
cont.runCont(f)
``` -->



既然类型 Cont[R, A] 是 (A => R) => R 的封装，那么前面我们提到的 CPS 函数能否转换成返回 Cont 的形式呢

```scala
val plus10CPS: Int => (Int => Unit) => Unit = a => f => f(a + 10)
val plus10Cont: Int => Cont[Unit, Int] = ?
```

答案是显然的，只需要用 plus10CPS 传入第一个参数的返回值构造 Cont 对象就行了

```scala
val plus10CPS: Int => (Int => Unit) => Unit = a => f => f(a + 10)

val x: Int = ...
val c2 = plus10CPS(x)
// val c2 = f => f(x + 10) // 另一种表示方法
val plus10Cont: Int => Cont[Unit, Int] = a => new Cont(c2)
```

||plus10CPS|plus10Cont|
|--|--|--|
|输入类型|Int|Int|
|返回类型|(Int => Unit) => Unit|Cont[Unit ,Int]|

上表是 plus10CPS 和 plus10Cont 的简单对比。mul2Cont 也可以如法炮制

```scala
val mul2Cont: Int => Cont[Unit, Int] = a => new Cont(f => f(a * 2))
```

反过来， Cont 形式的函数也可以得到 CPS 函数，我们仍然一步一步推导，首先可以写出 plus10CPSFromCont 的大致形式

```scala
plus10CPSFromCont: Int => (Int => Unit) => Unit = a => f => {
  ...
}
```
这里的 a 是 CPS 函数的第一个参数，并且恰好也是 plus10Cont 的第一个参数，所以

```scala
plus10CPSFromCont: Int => (Int => Unit) => Unit = a => f => {
  val c1: Cont[Unit, Int] = plus10Cont(a)
  ...
}
```
而 f 是 continuation，又刚好是 c1.runCont 的参数，于是

```scala
plus10CPSFromCont: Int => (Int => Unit) => Unit = a => f => {
  val c1: Cont[Unit, Int] = plus10Cont(a)
  c1.runCont(f)
}
```

简化一下

```scala
plus10CPSFromCont: Int => (Int => Unit) => Unit = a => f => plus10CPS(a).runCont(f)
```

接下来又来到了喜闻乐见的函数组合环节，我们需要把两个 Cont 函数组合成一个更大的函数

```scala
val plus10ThenMul2Cont: Int => Cont[Unit, Int] = ...
```

有了 CPS 函数组合的经验，我们只需要在此基础上略微改动，首先，由于返回值类型是 Cont，所以最终结果应该被用来新建 Cont 对象

```scala
val plus10ThenMul2Cont: Int => Cont[Unit, Int] = a => 
    new Cont(
        f => plus10CPS(a)(b => mul2CPS(b)(f))
    )
```

然后，plus10CPS 应该从 plus10Cont 得到，mul2CPS 应该从 mul2Cont 得到，这我们刚好学过，于是组合后的函数就为

```scala
val plus10ThenMul2Cont: Int => Cont[Unit, Int] = a => 
    new Cont(
        f => plus10Cont(a).runCont(b => mul2Cont(b).runCont(f))
    )
```

到这里，我们费劲心思，终于把两个返回 Cont 的函数组合成了一个更大的函数，其中意义又在哪里呢？当然不是为了花式的表达 

```haskell
\x -> 2 * (10 + x)
```

而是为了实现下面要介绍的 Cont Monad 的 bind (也就是 >>=) 函数

#### 实现 Cont Monad

类似于 Maybe, [] 这些类型，Cont 也是一个 Monad，在 haskell 中，它的声明如下

```haskell
instance Monad (Cont r) where 
    return :: a -> Cont r a
    return x = ...

    >>= :: m a -> (a -> m b) -> m b
    m >>= k = ...
```

其中 return 很好实现，它相当于当前计算为自身，也就是

```haskell
return x = Cont \f -> f x
-- 对比 plus10
plus10Cont x = Cont \f -> f $ x + 10
```

由于 return 是 scala 语言的关键字，所以在 scala 中我们用 \`return` 代替

```scala
def `return`[R, A](x: A): Cont[R, A] = new Cont(f => f(x))

// 当然也可以用 lambda 表达式定义
val `return`: A => Cont[R, A] = x => new Cont(f => f(x))
```

在 scala 中， bind (>>=) 函数的类型为 

```scala
def >>=[B](k: A => Cont[R, B]): Cont[R, B] = {
    ...
}
```

这里的参数 k 的类型和前面我们举的例子 plus10Cont，mul2Cont 是一样的，我们令

```scala
val x = 10
val c1 = plus10Cont(x)
val c3 = c1.bind(mul2Cont)
```

如果让 plus10Cont 自述它的行为

> 当前我所要进行的计算是 b = x + 10，但对我来说 x 仍是未知数，需要调用者提供，并且完成当前计算之后，我还需要一个 f 函数，把 b 传递给它，最终才能完成计算。

那么 c1 这个 Cont 表达的语义则是

> 当前我所要做的计算是 f(b)，b 对我来说是已知的，但是 f 还未知晓，需要调用者通过 runCont 接口提供。

如果我们调用 c1 的 runCont 方法，并且 

```scala
val f = x => println(x)
```

就相当于告诉它 

> hi! c1，我这里有个 continuation，名叫 f，现在我将它传给你，需要你运行它。

于是 c1 完成了计算。但当我们调用 c1 的 bind 方法时，事情就变得稍微复杂一点了，同 plus10Cont 一样，mul2Cont 的自述是

> 当前我所要进行的计算是 d = y * 2，但对我来说 y 仍是未知数，需要调用者提供，并且完成当前计算之后，我还需要一个 g 函数，把 d 传递给它，最终才能完成计算。

现在的问题是 y 的值是多少？处在 c1 的计算环境中，唯一的已知量是 b，所以毫无疑问 y = b，但是这里的 b 得处于 runCont 的环境下才能获得，所以在 bind 函数内部，我们需要运行它自身的 runCont

```scala
val some = g => runCont(b => {
    val cm = mul2Cont(b)
    cm.runCont(g)
})
```

这里我们显然不是 >>= 的返回类型，另一方面，g 也还不确定。为了得到 Cont 类型的返回值，和之前一样，我们需要显式地构造它，也就是

```scala
new Cont(h => ...)
```

这里的 h 是完成当前计算后需要传给 runCont 的参数，而 g 正是传给 cm 的 runCont 的，所以何不让 h = g，从而完成 bind 函数的实现

```scala
def bind(k: A => Cont[R, B]): Cont[R, B] = {
    new Cont(g => runCont(b => {
        val cm = k(b)
        cm.runCont(g)
    }))
}
```

简化一下 

```scala
def >>=[B](k: A => Cont[R, B]): Cont[R, B] = {
    new Cont(g => runCont(b => k(b).runCont(g)))
}
```

如果我们把前面讨论的两个 Cont 函数的组合方法拿过来和 >>= 进行对比

```scala
val plus10ThenMul2Cont: Int => Cont[Unit, Int] = a => 
    new Cont(
        f => plus10Cont(a).runCont(b => mul2Cont(b).runCont(f))
    )
```

上面的 plus10Cont(a) 是一个 Cont，在 Cont 内部的可以直接调用 runCont，如果把 plus10Cont(a) 省略掉再看

```scala
new Cont(
        f => runCont(b => mul2Cont(b).runCont(f))
    )
```

可以发现，这不就是 >>= 方法的函数体吗？所以看起来晦涩的 >>= 方法实际上表达的是两个连续操作的组合，只不过是在其中一个 Cont 内进行的，返回的也是 Cont。这样理解的话，haskell 中 Cont 的 >>= 也就一目了然的

```haskell
m >>= k = Cont $ \f -> runCont m (\a -> runCont k f)
```

下面我们给出 scala 版本的 Cont 实现

```scala
class Cont[R, A](cont: (A => R) => R){

  def >>=[B](k: A => Cont[R, B]): Cont[R, B] = {
    Cont(f => runCont(b => k(b).runCont(f)))
  }

  def runCont(f: A => R): R = cont(f)
}
    
object Cont {

  def `return`[R, A](a: A): Cont[R, A] = {
    Cont((f: A => R)=> f(a))
  }

  def apply[R, A](cont: (A => R) => R): Cont[R, A] = new Cont(cont)

}
```

简单使用一下

```scala
Cont.`return`[Unit, Int](10)
      .>>=(res => Cont.`return`(res * 2))
      .>>=(res => Cont.`return`(res / 3.0))
      .runCont(a => println(a))
```

很棒！这一节，我们介绍了用 Cont 封装计算的方法，在形式上，实现了把连续操作组合成一个更大操作的流式调用风格。但是如果中间操作抛出异常的话，整个过程就中断了，为了捕捉异常，我们需要在语句外加上 try catch 语法，这就属于干脏活了。为了更优雅的处理中间过程发生的异常，下面我们开始介绍 callcc。


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



<!-- 下面我们来到对 Cont 的讨论，其实 Cont 正是对 execCallback 的封装，封装方法为

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

经过以上讨论，我们给出 Cont 的完整形式 -->


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
<!-- 
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

运行代码，可以看到先打印出了 "end"，说明前面的连续处理是异步的，并且没有回调嵌套的问题。 -->

#### 使用 callcc 进行异常处理



#### 总结

<!-- 理解 Cont Monad，如果直接看源码的话，会显得相当困难，尤其是 bind 函数的实现，给人的感觉就是，虽然起作用，但是很懵。所以，我觉得先搞清楚 CPS 函数的组合，再到返回 Cont 版本的函数组合，能够从原理上讲清楚为什么是那样的。 -->




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