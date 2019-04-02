---
layout: post
title: Java 实现网络文件传输
description:
tags: Java NIO
---

在 Java 网络编程种，有时候需要传输一个文件到另一端，它的原理我们可以用一个很简单的过程来描述，大致如下所示

```sequence {theme="hand"}
Client -> Server: connect
Note right of Server: accept
Note left of Client: read bytes
Client -> Server: send bytes
Note right of Server: write to file
Server -> Client: OK
```


我们先简单实现一下

```java
//client end
//version 0.01
public class Client{
    public static void main(String[] args) throws IOException{
        var host = "127.0.0.1";
        var port = 12345;
        var socket = new Socket(host, port);
        var dout = new DataOutputStream(socket.getOutputStream);
        var fin = new FileInputStream(new File("/path/to/file"));
    }
}
```

```python {cmd="G:/ProgramData/Anaconda3/Scripts"}
print("hello code chunk")
```
