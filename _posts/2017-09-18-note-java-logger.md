---
layout: default
---

## 笔记：Java Logger

使用日志，首先就是要拿到日志对象，然后再指定日志文件存储路径嘛。不知道怎地，搞半天没在网上找到一篇关于怎么通过配置文件指定日志存储目录的文章。最后发现这篇博文 [Java Logging: Handlers](http://tutorials.jenkov.com/java-logging/handlers.html) 才搞定这个问题。

首先获取 Logger 实例对象

```java
Logger logger = Logger.getLogger("myLog");
```

这里的参数为日志名称，在其他地方如果需要此日志对象，可以通过名称来获取。然后使用 LogManager 来读取配置文件

```java
try {
    LogManager.getLogManager()
              .readConfiguration(new FileInputStream(new File("path/to/logger.properties")));
} catch (IOException e) {
    e.printStackTrace();
}
```

在配置文件中，需要指定许多项目，这里只涉及存储路径的问题

```
java.util.logging.FileHandler.level=INFO
java.util.logging.FileHandler.formatter=java.util.logging.SimpleFormatter
java.util.logging.FileHandler.pattern=pattern

handlers=java.util.logging.FileHandler
```

上面的值 "pattern" 是存储的模式，如果单纯的只是想设定一个存储路径，那么直接写个绝对路径就好了。如果只写一个文件名，那么日志将存储在项目的根目录下。

好了，就这么多。










