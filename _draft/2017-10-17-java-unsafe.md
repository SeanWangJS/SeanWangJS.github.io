---
layout: default
---

使用 Unsafe 对象获取数组长度
原理：数组对象的对象头由三个字段组成，即 Mark Word, 类型指针(class pointer), 数组长度
在64位 JVM 中 Mark Word 占用 8 字节，类型指针 占用 8 字节，于是为了得到获取数组长度字段，需要偏移 16 个字节
```java
int[] arr = new int[10];
int length = unsafe.getLong(arr, 16L);
```
但在开启了 JVM 指针压缩的情况下(在我的机器上默认是这样的，64位 JDK 9 版本)，类型指针只占用 4 字节，于是相应的获取数组长度字段修改为
```java
int length = unsafe.getLong(arr, 12L);
```
