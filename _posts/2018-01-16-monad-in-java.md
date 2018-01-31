---
layout: post
title: 在 Java 中实现 Monad
---

### 少量预备知识

考虑到范畴论是一门复杂而深邃的数学分支，这里我无意去冒昧探索，只是按照自己的理解介绍一些后续用得到的名词概念。

首先说 *范畴*，范畴是一个抽象的数学概念，它里面包含有一些对象，而这些对象依然是抽象概念，现在不必深究它究竟为何物，我们只知道，这些对象之间存在映射关系，这种关系在范畴论中有个专门的名称叫做 *态射*。例如有两个集合 $$S_1, S_2$$ 是范畴 $$\mathcal{C}$$ 的所谓的对象，它们之间存在函数关系 $$f: S_1 \to S_2$$，那么 $$f$$ 就是范畴 $$\mathcal{C}$$ 的一个态射。当然需要了解的是范畴里的对象不必是集合，态射也不必是函数。

![](/resources/2018-01-16-monad-in-java/category.png)
（一个范畴及其对象与态射的例子）

类似于集合或者群，两个范畴之间也显然存在映射关系，它不仅将一个范畴中的对象映射为另一个范畴的某个对象，而且还把态射也映射了过去，我们把这种范畴之间的映射称为 *函子*。

![](/resources/2018-01-16-monad-in-java/categories.png)
（范畴间的函子关系）

### 使用 Java 代码实现函子

（为了不和 Java 的 *对象* 这一概念冲突，下面我们用 *物件* 这一个词来代替范畴的对象）要用 Java 实现一些范畴论上面的概念，很明显我们需要在 Java 中找到能相对应的元素。举个简单的例子，我们可以将 String 类作为一个范畴的物件对象，而从 String 到 String 的态射可以用一个 lambda 表达式 Function<String, String> 来对应，这就构建了一个相当简单的范畴。

接下来我们需要再人为构建一个范畴用来作为前一个范畴的映射结果。一个简单的想法是定义一个类用来包装 String 的实例变量，我们把它暂且叫做 StringWrapper 吧。代码如下

```java
// version=0.0.1
class StringWrapper{
  public final String s;
  public StringWrapper(String s) {
    this.s = s
  }
  @Overide
  public String toString() {
    return s;
  }
}
```

也就是说 StringWrapper 是另一个范畴的物件。当然从 StringWrapper 到 StringWrapper 的态射可用 Function<StringWrapper, StringWrapper>表达。接下来该处理函子的问题了，由于 Java 的基本单元是类，所以我们把函子定义成类 Functor。函子的两个能力——将物件映射、将态射映射，可以使用 Functor 的两个方法来表达。代码如下

```java
// version=0.0.1
public class Functor{

  public StringWrapper map(String s) {
    return new StringWrapper(s);
  }
  pulbic Function<StringWrapper, StringWrapper> map(Function<String, String> f) {
    return wrapper -> new StringWrapper(f.apply(wrapper.s));
  }

}
```

下面是测试的例子

```java
@Test
public void test() {
  Functor functor = new Functor();
  StringWrapper sw = functor.map("hello");
  Function<StringWrapper, StringWrapper> ff = functor.map(s -> s + " functor");
  StringWrapper result = ff.apply(sw);
  System.out.println(result);
}
```

结果将打印出 hello functor。

这是一个相当朴素的例子，我们一套代码只能处理一个类型的物件，这显然是毫无效率可言的，下面我们使用泛型来重构一下代码

```java
// version=0.0.2
// wrapper
public class Wrapper<T> {
  public final T t;
  public Wrapper(T t) {
    this.t = t;
  }
  @Override
  public String toString() {
    return t.toString();
  }
}

//functor
public class Functor{
  public <T> Wrapper<T> map(T t) {
      return new Wrapper<>(t);
  }

  public <T> Function<Wrapper<T>, Wrapper<T>> map(Function<T, T> f){
      return wp -> new Wrapper<>(f.apply(wp.t));
  }
}
```

测试：

```java
@Test
public void test() {
  Functor functor = new Functor<>();
  Wrapper<String> hello = functor.map("hello");
  Function<Wrapper<String>, Wrapper<String>> ff = functor.map(s -> s + " world");
  Wrapper<String> result = ff.apply(hello);
  System.out.println(result);
}
```

![](/resources/2018-01-16-monad-in-java/categories-code.png)
（Java 版范畴映射的图形表示）

如果将前面的代码用示意图表示出来，再与纯粹的范畴映射图形进行比较会发现，如果我们使用函子 *F* 来作用于 $$S_1$$，得到的物件为 $$F(S_1)$$，然而在代码中，我们使用的是 Functor 的 map 方法，得到的却是 Wrapper 类型。这一差异就启发我们是否可以直接将 Functor 与 Wrapper 的功能合并，也就是说，Functor 既能作为包装类型，又具有函子的能力，这样代码具有更紧凑的形式。同时，我们取消对物件的 map 方法，代之以构造方法。代码如下

```java
// version=0.0.3
public class Functor<T> {
  public final T t;

  public Functor(T t) {
    this.t = t;
  }

  public Function<Functor<T>, Functor<T>> map(Function<T, T> f) {
    return func -> new Functor<>(f.apply(func.t));
  }

  public String toString() {
    return t.toString();
  }
}
```

测试:

```java
@Test
public void test() {
  Functor<String> functor = new Functor<>("hello");
  Function<Functor<String>, Functor<String>> ff = functor.map(s -> s + " world");
  Functor<String> result = ff.apply(functor);
  System.out.println(result);
}
```

在应用时，我们分别封装了基本范畴的一个物件对象 "hello" ，以及一个态射，获得了一个映射范畴的一个物件 functor，以及另一个态射 ff。然后将映射后的物件应用于后一个态射，最后得到结果。在这一过程中，我们显式地把后一个态射 ff 给了出来，但是从结果来看这一步并非是必要的，因为 Functor 本身是知道我们要处理的物件，于是可以在 Functor 内部应用态射，直接得到结果，从而省略返回态射 ff 那一步。重构代码如下

```java
// version=0.0.4
public class Functor<T>{
  public final T t;

  public Functor(T t) {
     this.t = t;
  }

  public Functor<T> map(Function<T, T> f) {
     //Function<Functor<T>, Functor<T>> ff = wrapper -> new Functor<>(f.apply(wrapper.t));
     //return ff.apply(this);
     // or equality
     return new Functor<>(f.apply(t));
  }

  public String toString() {
     return t.toString();
  }

}
```

测试：

```java
@Test
public void test(){
  Functor<String> result = new Functor<>("hello").map(s -> s + " world");
  System.out.println(result);
}
```

这种将态射在类 Functor 内部应用的行为让我们获得了一个额外却很重要的特性，那就是链式调用。它可以不断地将变换应用于 Functor 包裹的值，例如下面的代码所示

```java
Functor<Integer> r2 = new Functor<>(1).map(i -> i + 1)
                .map(i -> i * 2)
                .map(i -> i - 10);
System.out.println(r2);
```

我们前面的讨论，都是在考虑一个物件对象类型的前提下，基本范畴内的态射只有 $$f: T \to T$$。下面我们再考虑稍微复杂一点的情况，允许不同物件之间的态射，即 $$f:T\to R$$，仍然借助于 Wrapper 类，代码如下

```java
// version=0.0.5
class Wrapper<T> {
    public final T t;
    public Wrapper(T t) {
        this.t = t;
    }
    @Override
    public String toString() {
        return t.toString();
    }
}

public class Functor{
  public <T, R> Function<Wrapper<T>, Wrapper<R>> map(Function<T,R> f) {
    return wp -> new Wrapper<>(f.apply(wp.t));
  }
}
```

测试：

```java
@Test
public void test() {
    Functor functor = new Functor();
    Function<Wrapper<String>, Wrapper<Integer>> ff = functor.map(String::length);
    Wrapper<String> hello = new Wrapper<>("hello functor");
    Wrapper<Integer> result = ff.apply(hello);
    System.out.println(result);
}
```

再一次将 Wrapper 的能力整合进 Functor，并且不显式返回态射

```java
// version=0.0.6
public class Functor<T> {
  public final T t;

    public Functor(T t) {
        this.t = t;
    }

    public <R> Functor<R> map(Function<T, R> f) {
        return new Functor<>(f.apply(t));
    }

    public String toString() {
        return t.toString();
    }
}
```

测试：

```java
@Test
public void test() {
  Functor<Double> result = new Functor<>("hello functor")
                  .map(String::length)
                  .map(i -> i / 2.0);
  System.out.println(result);
}
```

这时我们的 Functor 类便能够在任意类型之间实施链式映射，极大地扩展了接口的泛用性。我们可以用 Functor 包裹任意类型的值，然后连贯地对其进行转换，最终得到我们需要的结果（当然也是包裹在 Functor 内部的）。

### 将函子抽象为接口

如果我们直接将函子定义为一个类，将会有很大的局限性。因为，如果其他类也想实现函子的功能就必须继承 Functor，由于 Java 不允许多重继承，所以对没有明显父子关系的类使用继承是一种不好的结构。显然，其他类只是想获得函子的能力，从而能更好地用于其他用途，所以更好的方式是将 Functor 作为接口

```java
interface Functor<T>{

  <R> Functor<R> map(Function<T, R> f);

}
```

下面我们可以实现这个接口，方法和之前一样

```java
public class FunctorImpl<T> implements Functor<T> {

  public T t;
  public FunctorImpl(T t){
    this.t = t;
  }

  public <R> Functor<R> map(Function<T, R> f) {
    return new FunctorImpl<>(f.apply(t));
  }

  @Overide
  public String toString() {
    return t.toString();
  }

}
```

测试：

```java

@Test
public void test() {
  FunctorImpl<Integer> functor = new FunctorImpl(10);
  System.out.println(functor.map(i -> i / 10.0));
}
//output: 1.0
```


待续。。


end


end
