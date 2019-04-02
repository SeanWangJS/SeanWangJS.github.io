---
笔记：如何在Spark 程序中修改 java.library.path
---

需要在spark程序中调用本地代码，发现老是加载不上动态库文件，于是在网上找了半天，试了各种方法，后来定位到应该是 java.library.path 的问题。但是尽管 source 了 export LD_LIBRARY_PATH ，还是不行。在程序中打印 java.library.path 依旧是老样子，搞到晚上十二点半正准备放弃，无意间发现 spark-env.sh 下面这句

```
if [ -n "$HADOOP_HOME" ]; then
  export LD_LIBRARY_PATH=:/usr/lib/hadoop/lib/native
fi
```

原来是这玩意儿给我覆盖了。。奇怪这种写法，难道不应该写成下面这样吗？

```
export LD_LIBRARY_PATH=:/usr/lib/hadoop/lib/native:$LD_LIBRARY_PATH
```

