---
layout: post
title: Jigsaw 项目--模块系统快速导引（中文翻译）
---

原文地址：[Project Jigsaw: Module System Quick-Start Guide](http://openjdk.java.net/projects/jigsaw/quick-start)。

本文为了开发者快速入门 Java 模块化而介绍几个简单的例子。

在例子中，文件路径使用斜杠，且用冒号分隔不同的路径。对于 Windows 开发者，应该使用反斜杠标志文件路径，使用分号作为文件分隔符。

* [Greeting](#greetings)
* [Greeting world](#gworld)
* [多模块编译](#multi-module)
* [打包](#packaging)
* [缺失 requires 或者缺失 exports](#missing)
* [服务](#service)
* [链接器](#linker)

### <span id="greetings">Greeting</span>

第一个例子是一个打印出 "Greetings!" 的简单模块，这里命名为 com.greetings. 模块包含两个源文件，即模块声明(module-info.java)和主类。

根据惯例，源码所在的文件夹以模块名称命名。

```
	src/com.greetings/com/greetings/Main.java
	src/com.greetings/module-info.java

	$ cat src/com.greetings/module-info.java
	module com.greetings { }

	$ cat src/com.greetings/com/greetings/Main.java
	package com.greetings;
	public class Main {
		public static void main(String[] args) {
			System.out.println("Greetings!");
		}
	}
```

使用下面的命令，将源码编译到目标文件夹 mods/com.greetings

```
	$ mkdir -p mods/com.greetings

	$ javac -d mods/com.greetings \
		src/com.greetings/module-info.java \
		src/com.greetings/com/greetings/Main.java
```

然后运行示例

```
	$ java --module-path mods -m com.greetings/com.greetings.Main
```

其中 --module-path 指定了模块路径，其值为一个或者多个包含模块的文件夹。-m 选项指定了主模块，斜杠后面的值是模块主类的名字

### <span id="gworld">Greetings world</span>

下面来看第二个例子，它更新了模块声明，宣布该模块依赖于另一个模块 org.astro 。而模块 org.astro 导出了 org.astro 的 API 包。

```
    src/org.astro/module-info.java
    src/org.astro/org/astro/World.java
    src/com.greetings/com/greetings/Main.java
    src/com.greetings/module-info.java

    $ cat src/org.astro/module-info.java
    module org.astro {
        exports org.astro;
    }

    $ cat src/org.astro/org/astro/World.java
    package org.astro;
    public class World {
        public static String name() {
            return "world";
        }
    }

    $ cat src/com.greetings/module-info.java
    module com.greetings {
        requires org.astro;
    }

    $ cat src/com.greetings/com/greetings/Main.java
    package com.greetings;
    import org.astro.World;
    public class Main {
        public static void main(String[] args) {
            System.out.format("Greetings %s!%n", World.name());
        }
    }
```

这两个模块将被逐个编译。使用 javac 编译模块 com.greetings 时，需要指定模块路径，才能找到 org.astro 模块和它导出包中的类型。

```
$ mkdir mods/org.astro mods/com.greetings

    $ javac -d mods/org.astro \
        src/org.astro/module-info.java src/org.astro/org/astro/World.java

    $ javac --module-path mods -d mods/com.greetings \
        src/com.greetings/module-info.java src/com.greetings/com/greetings/Main.java
```

和第一个例子一样，使用下面命名运行

```
$ java --module-path mods -m com.greetings/com.greetings.Main
    Greetings world!
```

### <span id="multi-module">多模块编译</span>

在前面的例子中，com.greetings 和 org.astro 是分开编译的。当然也可以使用一条 javac 命令来编译多个模块:

```
    $ mkdir mods

    $ javac -d mods --module-source-path src $(find src -name "*.java")

    $ find mods -type f
    mods/com.greetings/com/greetings/Main.class
    mods/com.greetings/module-info.class
    mods/org.astro/module-info.class
    mods/org.astro/org/astro/World.class
```

### <span id = "packaging">打包</span>

到目前为止，模块编译是在文件系统上操作的。为了传输和部署，将模块打包成模块化的 jar 会更方便。一个模块化的 Jar 其实就是在普通的 Jar 的顶级目录下加入了 module-info.class 文件。下面的例子在 mlib 目录下创建了 org.astro@1.0.jar 和 com.greetings.jar.

```
$ mkdir mlib

$ jar --create --file=mlib/org.astro@1.0.jar --module-version=1.0 -C mods/org.astro

$ jar --create --file=mlib/com.greetings.jar --main-class=com.greetings.Main -C mods/com.greetings

$ ls mlib/com
com.greetings.jar org.astro@1.0.jar
```

在这个例子中，打包模块 org.astro 指定了其版本号为 1.0。而模块 com.greetings 则指定了其主类为 com.greetings.Main。现在我们可以无需指定主类来运行模块 com.greetings

```
$ java -p mlib -m com.greetings
Greetings world!
```

上面的命令使用 -p 来代替 --module-path 从而缩短命令长度。

jar 工具新增了不少选项 (见 jar -help)，其中之一是打印出模块 Jar 的模块声明。

```
$ jar --describe-module --file=mlib/org.astro@1.0.jar
org.astro@1.0 jar:file:///d/mlib/org.astro@1.0.jar/!module-info.class
exports org.astro
requires java.base mandated
```

### <span id = "missing">缺失 requires 或者缺失 exports</span>

















