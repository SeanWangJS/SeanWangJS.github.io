---
layout: default
---

## 位运算： 边界检查

今天在看 nio.ByteBuffer 源码的时候发现一个函数
```java
public ByteBuffer get(byte[] dst, int offset, int length);
```
它的作用是将缓冲区数据读入 dst 数组，读入 dst 的起始偏移量为 offset，读取长度为 length。显然这几个参数必须满足多个条件：

1. offset >= 0
2. length >= 0
3. offset + length 小于 dst 的大小

为了保证传入参数的正确性，需要进行参数越界检查

```java
checkBounds(offset, length, dst.length);
```

下面是 checkBounds 的实现

```java
static void checkBounds(int off, int len, int size) {
    if ((off | len | (off + len) | (size - (off + len))) < 0)
            throw new IndexOutOfBoundsException();
}
```
也就是说，只要 off , len , off + len, size-(off+len) 中的任意一个量小于 0，经过或运算之后其符号位应该为 1，即结果小于 0，这时就表示传入的参数没有通过边界检查。
