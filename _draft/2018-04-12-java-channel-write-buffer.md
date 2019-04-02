---
layout: post
title: 当向 Java 通道中写入字节的时候发生了什么？
description:
tags: Java源码分析 NIO
---

之前在写网络程序的时候遇到一个bug，简单来说就是客户端向服务端发送数据，一开始想都没想就直接来了个

```java
ByteBuffer buffer = ...;
socketChannel.write(buffer);
```

在运行的时候，发现服务端老是收不到数据，就一直阻塞在接受数据的状态，而客户端这边早早地就退出了。一开始以为是网速的问题，于是在发送端这边加了延时，发现也没用。然后又怀疑是否是发送缓冲区或者接收缓冲区溢出的原因，但即便发送很少的数据或者调高缓冲区也不行。后来经过一系列折腾之后发现，在客户端中，socketChannel 写入 buffer 后根本没有发送出去，而只是保存了起来，当程序逻辑运行完了之后也就结束了，结果服务端那边还在苦苦等待。

意识到这一点之后，瞬间反应过来 ByteBuffer 还有个方法叫 hasRemaining() ，于是将上述发送过程改成了

```java
ByteBuffer buffer = ...;
while(buffer.hasRemaining()) {
  socketChannel.write(buffer);
}
```
就瞬间没毛病了。

















end
