---
layout: post
title: Java 中的动态代理
tags: Java 动态代理
description: 介绍了动态代理的使用场景，实现了简单的动态代理框架，深入分析了 Java 动态代理源码。
modify_date: 2018-03-24
---

### 动态代理需求的出现

代理是一种设计模式，其作用是对原方法进行增强，并且不具侵入性。比如一个接口提供某个服务

```java
public interface Service{
  public void serve(String[] args);
}
```

然后有一个类实现此服务

```java
public class ServiceImpl implement Service{
  @Override
  public void serve(String[] args){
    System.out.println("serving..." + Arrays.toString(args));
  }
}
```

但如果我们想在正式调用服务之前或之后做一些额外工作，比如参数校验、资源关闭等等。若不使用代理，那么就只能修改源码实现，这就破坏了原服务方法的纯粹性，更糟糕的是，如果这个服务来自于第三方库，那就连源码都修改不了了。

这时候可以使用代理的方式来增强原方法的行为，首先定义一个代理类实现 Service 接口，然后注入原来的服务类 ServiceImpl 实例，这样代理类就拿到了原服务类的引用。并且由于代理类本身就是一个 Service，所以调用者在用代理类的时候感觉就和在使用原服务类一样。

```java
public class ServiceProxy implements Service{
  private Service service;

  public ServiceProxy(Service service) {
    this.service = service;
  }

  @Override
  public void serve(String[] args) {
    before();
    service.serve(args);
    after();
  }

  private void before(){
    System.out.println("do something before serving...");
  }

  private void after() {
    System.out.println("do something after serving...");
  }
}
```

但是通常程序员都很懒，总是想从现有方法中抽象出通用模式，从而更少地编码。我们看到上面的代理方法，为了在 serve() 方法执行前后分别加入 before() 和 after() 方法，手动写了一个 ServiceProxy 类。如果要代理的类有很多个，那么就需要为每个类写一个对应的代理类，这显然是不够高效的。于是我们就想，能不能一股脑儿给出要代理的类实例、方法、方法参数和增强代码，然后直接从某个工厂方法中得到代理类对象呢？类似于下面的方法

```java

Object getProxy(Object obj, Method method, Object[] args, TriFunction<Object, Method, Object[], Object> enhancement)

```

其中 obj 是被注入代理类的实例，method 是要被代理的方法，args 是方法参数列表，最关键的是 enhancement ，它是一个 lambda 表达式，内部写入了增强内容，前三个泛型变量分别就是 getProxy 方法的前三个参数，第四个泛型参数是 method 的返回值，这样我们就为增强代码提供了必要的信息，它的增强写法类似下面这样

```java
TriFunction<Object, Method, Object[], Object> enhancement = (obj, method, args) -> {
  // ... do something before
  Object result = method.invoke(obj, args);
  // ... do something after

}
```

当然由于 lambda 表达式的无状态性，有可能会限制增强能力，这时我们可以考虑一个重载方法来获取代理类

```java
Object getProxy(Object obj, Method method, Object[] args, Enhancement enhancement);

```

这个 Enhancement 类里面就定义了增强方法，并且还可以保存某些状态以更灵活的方式进行增强。比如它可以被写成下面这样

```java
public class Enhancement{

  private Object obj;
  private Method method;
  private Object[] args;

  public Enhancement(Object obj, Method method, Object[] args) {
    this.obj = obj;
    this.method = method;
    this.args = args;
  }
  public Object run() {
    // do something before
    Object result = method.invoke(obj, args);
    // do something after;
  }
}
```

现在我们的问题就是如何生成代理对象了，一般来讲，要创建对象直接 new 一个就行了，但是现在我们连要创建对象的类名字都不知道，当然也不可能使用反射来创建。事实上，我们要创建的对象它的类现在还不存在，于是就需要自己动手，丰衣足食了。首先考虑一下这个代理类的名称，其实并不重要，因为我们不会在外部代码中创建，只需要获取它的对象实例就行了，但为了不与其他使用同样方法生成的类重名，我们可以给它一个序号，即类似于

```java
public class Proxy1 implements $<Interface>${
  public Enhancement enhancement;
  public Proxy$1(Enhancement enhancement) {
    this.enhancement = enhancement;
  }
  @Override
  public Object $<method>$($<params>$){
    enhancement.run();
  }
}
```

由于代理类要实现与被代理类相同的接口，但是这个接口现在还不知道，所以我们先用一个占位符 \$\<Interface>$ 来临时代替，同样被用作占位符的还有 \$\<method>$ 表示方法名称和 \$\<params>$ 表示参数列表。从代理类方法调用的实现来看，并没有用到传入的任何参数，但为了使 Proxy1 看起来确实实现了原接口，补全参数列表是必须的，哪怕仅仅是名称匹配上了。

从代理类 Proxy1 的形式来看，这更像是一个模板，所以实际上，我们要做的就是根据被代理类的信息，根据上述模板手动构造出代理类的源码，然后使用 Java 动态编译以及动态加载技术获取代理类实例，这就是动态代理的实现。

### 实现自己的动态代理

为了更深刻地理解动态代理技术，我们按照刚才的思路实现一个工具，用于生成代理类，当然具体内容可能和前面不同，因为那只是一些设想，编程过程中可能会遇到一些问题，需要进行修正。首先来看一下代理类的模板

```java
import java.lang.reflect.Method;
$<import>$
public class $<className>$ implements $<interfaceName>$ {
  private Object obj;
  private Method method;
  private Enhancement enhancement;
  public $<className>$(Enhancement enhancement, Object obj) {
    this.enhancement = enhancement;
    this.obj = obj;
    this.method = method;
  }

  @Override
  public $<returnType>$ $<methodName>$($<paramsList>$){
    $<returnWord>$ enhancement.run(obj, method $<args>$);
  }

}
```

在上述模板中，各个要替换文本的含义如下

|模板文本|含义|
|:--|:--|
|\$\<import>$|导入包|
|\$\<className>$|代理类名称|
|\$\<interfaceName>$|实现接口名称|
|\$\<returnType>$|返回类型：void or Object|
|\$\<methodName>$|代理方法名称|
|\$\<paramsList>$|代理方法参数列表|
|\$\<returnWord>$|返回关键字： 空 or return|
|\$\<args>$|参数对象数组名称：空 or new Object[]{...}|

由于被代理方法可能有参数也可能没有参数，所以 Enhancement 可以使用两个接口方法来分别处理这两种情况

```java
interface Enhancement{

  Object run(Object obj, Method method, Object agrs);

  Object run(Object obj, Method method);

}
```

当被代理的方法有返回值的时候，可以直接在 run() 方法中返回，而当返回关键字为 void 时，\$\<returnWord>$ 会被空字符串替换，这时 run() 方法的实现可以返回任何内容。所以上面的设计能够同时应用于有返回值的方法和无返回值的方法。

为了生成代理类的源码，我们只需要简单的对模板进行文本替换，这些替换文本都应该能由已知信息获得。而这些已知信息应该作为 getProxy() 的参数被传入，所以让我们定义工厂类 ProxyFactory 以及工厂方法 getProxy()

```java
public class ProxyFactory{

  public static Object getProxy(...){
    ...
  }

}
```

然后来看看这个 getProxy() 方法的参数列表应该有些什么内容，我们总结如下

|参数|类型|名称|说明|
|:--|:--|:---|:--|
|被代理类接口|Class<?>|interf|代理类将实现该接口|
|被代理类实例|Object|obj|注入代理类|
|被代理方法|Method|method|需要在代理类中增强|
|增强内容|Enhancement|enhancement|用于增强原方法|


通过上述信息，我们可以获得接口名称

```java
  String interfaceName = interf.getSimpleName();
```

获得导入包信息

```java
  String importString =  "import " +interf.getPackageName() + "." + interf.getSimpleName() + ";";
}
```

获得返回信息

```java
  String returnType = "void".equals(method.getReturnType().getSimpleName()) ? "void" : "Object";
```

获得代理方法名称

```java
  String methodName = method.getName();
```

获得返回关键字

```java
  String returnWord = "void".equals(returnType) ? "" : "return";
```

接下来还有代理类名称以及代理方法参数列表，这里我们希望借助于一个数字增长器

```java
class Increment{
  private static int num = 0;
  public static int get(){
    return num++;
  }
  public static void clear(){
    i = 0;
  }
}
```

这样，代理类名称就为

```java
  String className = "Proxy" + Increment.get();
```

参数列表

```java
List<String> temp = new ArrayList<>();
String args = "";
String paramsList = "";
if(method.getParameterTypes().length != 0) {
  paramsList = Stream.of(method.getParameterTypes()).map(clazz -> {
      String argName = "arg" + Increment.get();
      temp.add(argName);
          return clazz.getSimpleName() + " " + argName;
      })
      .collect(Collectors.joining(", "));
  args = ", new Object[]{" +temp.stream().collect(Collectors.joining(", ")) + "}";
        }
```

然后对模板进行文本替换获得代理类源码，使用 Java 动态编译生成字节码，最后定义类加载器并加载字节码，于是 getProxy 的完整实现如下

```java
public static Object getProxy(Class<?> interf, Object obj, Method method, Object[] args, Enhancement enhancement) {
  List<String> temp = new ArrayList<>();
  String interfaceName = interf.getSimpleName();
  String className = "Proxy" + Increment.get();
  String args = "";
  String paramsList = "";
  if(method.getParameterTypes().length != 0) {
      paramsList = Stream.of(method.getParameterTypes()).map(clazz -> {
          String argName = "arg" + Increment.get();
          temp.add(argName);
          return clazz.getSimpleName() + " " + argName;
      })
              .collect(Collectors.joining(", "));
      args = ", new Object[]{" +temp.stream().collect(Collectors.joining(", ")) + "}";
  }
  Increment.clear();
  String importString =  "import " +interf.getPackageName() + "." + interf.getSimpleName() + ";";
  String methodName = method.getName();
  String returnType = "void".equals(method.getReturnType().getSimpleName()) ? "void" : "Object";
  String returnWord = "void".equals(returnType) ? "" : "return";

  String template = Template.getProxyTemplate();

  template = template.replace("$<import>$", importString)
                      .replace("$<interfaceName>$", interfaceName)
                      .replaceAll("\\$<className>\\$", className)
                      .replace("$<paramsList>$", paramsList)
                      .replace("$<method>$", methodName)
                      .replace("$<returnType>$", returnType)
                      .replace("$<returnWord>$", returnWord)
                      .replace("$<args>$", args);

  DynamicComplier complier = new DynamicComplier();
  complier.compile(className+".java", template);

  ClassLoader classLoader1 = new ClassLoader() {
      protected Class<?> findClass(String name) {
          byte[] byteCode = complier.get(name);
          return defineClass(name, byteCode, 0, byteCode.length);
      }
  };
  Object proxy = null;
  try {
      proxy = classLoader1.loadClass(className)
              .getDeclaredConstructor(Enhancement.class, Object.class, method.getClass())
              .newInstance(enhancement, obj, method);
  } catch (InstantiationException | NoSuchMethodException | ClassNotFoundException | InvocationTargetException | IllegalAccessException e) {
      e.printStackTrace();
  }

  return proxy;
}
```

其中的 DynamicComplier 是一个 Java 运行时编译器实现，在[这篇文章](/2018/03/13/java-runtime-compile.html)中我们有细致讨论。接下来我们用一下这个玩具般的动态代理框架

```java
@Test
public void test() throws NoSuchMethodException {
        ServiceImpl serviceImpl = new ServiceImpl();
        Enhancement enhancement = new Enhancement() {
            @Override
            public Object run(Object obj, Method method, Object[] args) {
                System.out.println("before");
                try {
                    method.invoke(obj, args);
                } catch (IllegalAccessException | InvocationTargetException e) {
                    e.printStackTrace();
                }
                System.out.println("after");
                return null;
            }
            @Override
            public Object run(Object obj, Method method) {
                return null;
            }
        };
        Service serve = (Service)ProxyFactory.getProxy(Service.class, serviceImpl,
                serviceImpl.getClass().getMethod("serve", String[].class), enhancement);
        serve.serve(new String[]{" param1 ", " param2 "});

}
/*output:
before
serving...[ param1 ,  param2 ]
after
*/
```

还不错，但是缺点很明显，比如由于模板的局限性，现在还不能代理有多个函数的类，下面我们开始研究 Java 标准库中给出的实现，看看有什么高超的地方。

### Java 中动态代理源码分析

我们先来看一下 Java 标准库中的动态代理用法

```java
@Test
public void test2() {
    ServiceImpl serviceImpl = new ServiceImpl();
    InvocationHandler handler = (proxy, method, args) -> {
        System.out.println("before");
        method.invoke(serviceImpl, args);
        System.out.println("end");
        return null;
    };
    Service service = (Service)Proxy.newProxyInstance(serviceImpl.getClass().getClassLoader(),
            serviceImpl.getClass().getInterfaces(),
            handler);
    service.serve(new String[]{" param1 ", " param2 "});
}
/*output:
before
serving...[ param1 ,  param2 ]
after
*/
```

可以发现，这里的代码也分为两个部分，第一部分是定义 InvocationHandler ，在它的 invoke 方法中进行了方法增强，和我们之前的 Enhancement 类似，但所不同的是它没有针对可能的带参数方法和不带参数方法分别写接口，而直接使用一个方法统一实现。第二部分则类似前面的代理工厂方法，它给的参数为被代理类的类加载器、被代理类的所有实现接口以及 InvocationHandler 对象。

现在我们直接进入 Proxy 的 newProxyInstance() 方法，里面最主要的就两句

```java
Constructor<?> cons = getProxyConstructor(caller, loader, interfaces);
return newProxyInstance(caller, cons, h);
```

上面第二行的 newProxyInstance 方法只有一行重要内容

```java
return cons.newInstance(new Object[]{h});
```

即通过构造器实例，以及参数 h 返回代理类实例，而这个构造器的生成方法主要在 getProxyConstructor() 方法中。在 getProxyConstructor() 里面，有一个条件语句确定接口数量，为单接口进行了优化，我们先看单接口处理分支，其返回语句为

```java
return proxyCache.sub(intf).computeIfAbsent(
                loader,
                (ld, clv) -> new ProxyBuilder(ld, clv.key()).build()
            );
```

从语义上来看，该句首先在缓冲区中查找接口 intf 的代理类构造器，如果找到了，则直接返回，否则生成。

可以看到，代理类是通过一个 ProxyBuilder 构造的，在这个构造器的 build() 方法中，有两句关键代码

```java
Class<?> proxyClass = defineProxyClass(module, interfaces);
final Constructor<?> cons;
try {
  cons = proxyClass.getConstructor(constructorParams);
} catch (NoSuchMethodException e) {
  throw new InternalError(e.toString(), e);
}
```

第一句是通过给定的模块和接口生成代理类的类类型，第二句是返回代理类的构造方法，它的构造参数为 constructorParams ，其声明为

```java
private static final Class<?>[] constructorParams =  { InvocationHandler.class };
```

也就是说，代理类的构造方法接受一个 InvocationHandler 的实例，这和我们前面自己的代理类构造方法中，接受一个 Enhancement 实例类似。接下来进入 defineProxyClass() 方法，该方法只接受一个模块，和被代理类的接口集合，从这里我们可以看到，Java 里面的动态代理将代理类的生成和方法增强分开处理。其中的生成方法只需要被代理类的接口信息即可，着实要比我们的逻辑高明一些。（但是从这一点我们也发现了一个问题，那就是如果被代理类实现了抽象类中的抽象方法，那么将无法正确生成代理类，因为传入参数中没有抽象类的任何信息）

在 defineProxyClass() 方法中，首先进行了一些检查，它的字节码生成语句为

```java
byte[] proxyClassFile = ProxyGenerator.generateProxyClass(
                    proxyName, interfaces.toArray(EMPTY_CLASS_ARRAY), accessFlags);
```

类加载语句为

```java
Class<?> pc = UNSAFE.defineClass(proxyName, proxyClassFile,
              0, proxyClassFile.length, loader, null);
```

代理类的字节码生成由 ProxyGenerator 的 generateProxyClass() 方法完成，这是一个静态方法，里面主要是建立 ProxyGenerator 对象，并调用实例方法 generateClassFile()

```java
ProxyGenerator gen = new ProxyGenerator(name, interfaces, accessFlags);
final byte[] classFile = gen.generateClassFile();
```

下面我们来到了最关键的字节码生成方法 generateClassFile() ，这里主要进行了三步操作，注释里写得很明确。第一步是为所有接口方法，以及三个特殊的 Object 方法 toString()，hashCode() ， equals() 生成 ProxyMethod 对象

```java
/* ============================================================
* Step 1: Assemble ProxyMethod objects for all methods to
* generate proxy dispatching code for.
*/
addProxyMethod(hashCodeMethod, Object.class);
addProxyMethod(equalsMethod, Object.class);
addProxyMethod(toStringMethod, Object.class);

for (Class<?> intf : interfaces) {
  for (Method m : intf.getMethods()) {
    addProxyMethod(m, intf);
  }
}
```

第二步，生成字段表集合以及方法表集合

```java
methods.add(generateConstructor());

for (List<ProxyMethod> sigmethods : proxyMethods.values()) {
  for (ProxyMethod pm : sigmethods) {

  // add static field for method's Method object
  fields.add(new FieldInfo(pm.methodFieldName,
                        "Ljava/lang/reflect/Method;",
                         ACC_PRIVATE | ACC_STATIC));

  // generate code for proxy method and add it
  methods.add(pm.generateMethod());
  }
}

methods.add(generateStaticInitializer());
```

第三步，直接向 ByteArrayOutputStream 实例写入代理类字节码，这里严格按照 Java 字节码文件的结构，向其中写入了 4 字节的魔数 0xCAFEBABE ，2 字节的次版本号，2 字节的主版本号，常量池，2 字节访问标记，2 字节类索引，2 字节父类索引，以及接口集合，字段集合，方法集合等等，由于代理类不需要属性表，所以属性数量写 0。

可以看到，不同于我们之前通过代理类模板生成代理类源码，然后调用动态编译接口生成字节码，在这里 JDK 直接拼装出了字节码。

到这里我们就大致梳理了一下 Java 中的动态代理实现过程，还有一些细节的地方没有涉及，以后有机会再来讨论。
