---
layout: post
title: Java HashMap 源码分析 01
tags: Java 源码分析
description: 简单介绍了哈希表的概念，详细分析了 Java HashMap 的 put 方法和 get 方法源码实现
---

本系列旨在深入研究 Java HashMap 的底层实现，在 JDK 9 中，HashMap 是一个拥有两千四百多行的大类，为了实现易用，强大，并且可靠的哈希表，里面有相当多的内容，希望通过这一系列文章，能对 HashMap 的实现有深入的理解。

### 哈希表

在具体讨论 HashMap 之前，我们先来看看哈希表这种数据结构。我们知道，数据结构是存储数据的方式，例如数组、链表这些简单的结构，而不同的数据结构对数据访问、查找、新增、删除、修改等操作的开销则存在差异，并且这种差异可能会非常巨大，于是使用不同的数据结构进行同样的编程活动，其性能差别也是非常大的，所以对数据结构的合理选择对程序的性能就至关重要。

像数组这种结构，在不考虑其容量大小的情况下，其数据访问、修改的时间复杂度为常数级 \\(O(1)\\)，但是新增、删除这两种操作，由于涉及到移动数组元素来开辟新的空间，或者填补空白区，就显得不那么高效了，时间复杂度达到了 \\(O(n)\\)。同样地，在一个数组中查找特定数据需要遍历操作，其时间复杂度也是 \\(O(n)\\)。

而为了获得各种操作都能在常数时间内完成的高性能数据结构，哈希表便应运而生了。要理解哈希表解决了什么问题，我们先来看看数组这种结构遇到了什么问题。在数组中，当我们知道某个数据的索引的时候，获得该数据只需一步操作，这就是**访问**，而如果不知道这个索引，那就必须一个一个的去找，这就是**查找**。所以为了高效地获得数据，我们必须知道这个数据在数组中的索引号。但是在数组中，索引号只能是像 0，1，2，3，4...这类数字，如果我们想存储某个人的信息，将其名字作为索引，那数组显然就无能为力了。但请记住那句名言

> 如果某个系统无法胜任某种需求，那么请在该系统上增加一层。

在这里的意思就是，我们可以将数组作为底层结构，在它上面添加一层逻辑，比如像名字这样的字符串无法作为数组索引，那么我们将该字符串映射到非负整数上，然后在用这个数作为索引号，不就OK了吗？这个映射就叫做**哈希函数**，映射后的数就叫**哈希码**，被映射的字符串就叫做**键**(key)，或者叫**关键字**，存储的数据就叫做**值**(value)，它们合起来被称为**键值对**，这一上层结构就被称为**哈希表**。

于是，在存储和读取数据之前，我们都先将key映射成数字，然后再与底层的数组打交道。这样一来，存储与读取数据的时间复杂度就都变成了 \\(O(1)\\)。同时还解决了删除数据的问题，因为我们根本不用移动数据，只需将该位置的值标记为空就行了。

![](/resources/2018-05-05-java-hashmap-analysis-01/hash.png)
<figcaption>哈希表的存储过程</figcaption>

当然，在很多时候，取决于哈希函数的选择，我们可能会将不同的key 映射成相同的数字，那么这时在数组上的位置就产生了冲突，这种情况被称为**哈希碰撞**，就像两个不同的数据在数组的同一个索引撞上了。我们当然不能直接覆盖原来的数据，否则就出错了，一个解决办法是，在该索引位置存储一个链表，当新数据到达该索引位置时，直接添加到链表后面。当访问数据时，除了要计算哈希码，还要在该位置进行链表搜索，这就不可避免地将时间复杂度增高了，甚至最坏的情况达到了 \\(O(n)\\) （此时所有key都被映射到了同一个哈希码）。但在非极端情况下，这种方案还是有效的，在JDK 8 中，如果该链表的长度超过某一阈值，则将其转换成红黑树，进一步降低了时间复杂度。

![](/resources/2018-05-05-java-hashmap-analysis-01/hash-modify.png)
<figcaption>使用链表修改后的哈希表存储过程</figcaption>


### HashMap 的键值对存储过程

Java 中的 HashMap 便是对上一节我们介绍的哈希表的具体实现，它的用法十分简单

```java
HashMap<String, Integer> map = new HashMap<>();
map.put("Alice", 10);
System.out.println(map.get("Alice"));
```

现在我们深入到 map 的 put 实现中，看看数据究竟是怎么存储起来的。找到 put 源码

```java
 public V put(K key, V value) {
        return putVal(hash(key), key, value, false, true);
    }
```

具体实现通过 putVal 方法，这里的 hash(key) 便是在将键映射成整数，关于哈希码有一套完整的理论，这里我们不展开讲，具体讨论请移步。。。。。，现在我们只需要知道通过 hash() 方法，将 key 转换成了一个整数，下面我们来逐句分析 putVal 方法中的代码

```java
Node<K,V>[] tab; Node<K,V> p; int n, i;
if ((tab = table) == null || (n = tab.length) == 0)
    n = (tab = resize()).length;
```

这一句是说，如果 tab 为空，或者它的长度等于0， 那么用 resize() 方法初始化一下，并将数组长度赋予变量n。这里的临时变量 tab 来源于实例变量 table，而这就是我们前面一直提到的底层存储数组，从源码中对它的注释来看

>/**
     * The table, initialized on first use, and resized as
     * necessary. When allocated, length is always a power of two.
     * (We also tolerate length zero in some operations to allow
     * bootstrapping mechanics that are currently not needed.)
     */

该数组在第一次使用时才被初始化，正是我们刚才看到的代码，并且在必要的时候重新分配长度，什么是“必要的时候”我们后面再讨论，还有它的长度永远都是 2 的幂。 resize() 中的内容我们也暂时略过，只需要知道，它将 table 的长度变化到合适的大小。我们继续

```java
if ((p = tab[i = (n - 1) & hash]) == null)
    tab[i] = newNode(hash, key, value, null);
else{
    ...
}
```

这里的 i = (n-1) & hash 就揭示了数组索引是如何得到的，原来是数组长度减一，再和刚才计算的 hash 值做 & 运算。当这个位置上没有元素的时候，就生成一个新的 Node，并放到该位置。从这里我们也看到，Node对象存储了键值对的基本信息。而当此索引位置上的元素不为空时，我们进入 else 分支

```java
Node<K,V> e; K k;
if (p.hash == hash &&((k = p.key) == key || (key != null && key.equals(k))))
    e = p;
```

这一句是在判断，我们要插入的关键字是否和原本的关键字相同，如果相同，则说明这是一次更新操作，只需要简单地覆盖即可，这里的实现是先用临时变量 e 来存储原来的 Node。否则

```java
else if (p instanceof TreeNode)
    e = ((TreeNode<K,V>)p).putTreeVal(this, tab, hash, key, value);
```

如果该位置上原本有一个 TreeNode 类型的对象，则将键值对存储到这个结构中（这里的 TreeNode 便是红黑树节点，我们前面提到过，为了提高性能，从 JDK 8 开始，红黑树成为了哈希碰撞之后的解决方案之一）。而如果 p 也不是红黑树节点，那么就意味着这个位置存储的是链表

```java
else {
    for (int binCount = 0; ; ++binCount) {
        if ((e = p.next) == null) {
            p.next = newNode(hash, key, value, null);
            if (binCount >= TREEIFY_THRESHOLD - 1) // -1 for 1st
                treeifyBin(tab, hash);
                    break;
        }
        if (e.hash == hash && ((k = e.key) == key || (key != null && key.equals(k))))
            break;
        p = e;
}
```

遍历这个链表，如果发现相同的关键字（for循环中第二个if语句）则退出循环，并且临时变量 e 已经存储了旧 Node，否则添加到链表尾部，并且如果链表的长度超过了设定阈值 TREEIFY_THRESHOLD ，则将该链表转换成红黑树。然后

```java
if (e != null) { // existing mapping for key
    V oldValue = e.value;
    if (!onlyIfAbsent || oldValue == null)
        e.value = value;
    afterNodeAccess(e);
    return oldValue;
}
```

如果 e 不为空，则说明我们这是一次更新操作，只需要将 e 的 value 属性设置为新值即可，然后返回旧值。最后

```java
++modCount;
if (++size > threshold)
    resize();
afterNodeInsertion(evict);
return null;
```

如果运行到这里，则说明这是一次新增操作，需要对 HashMap 的修改次数进行计数（为了并发控制检查），并且检查底层数组的容量是否该更新。以上就是 HashMap 的 put 方法大致内容，更细节的部分，比如 resize() 函数的实现、红黑树的插入，从链表生成红黑树的过程等等我们将在以后讨论。

### HashMap 的 get 过程

从 HashMap 中根据关键字取出数据用的是 get() 方法

```java
public V get(Object key) {
    Node<K,V> e;
    return (e = getNode(hash(key), key)) == null ? null : e.value;
}
```

这里，它先是用 getNode() 方法获得原始的 Node 对象，然后再返回其 value 属性，在 getNode() 中

```java
Node<K,V>[] tab; Node<K,V> first, e; int n; K k;
if ((tab = table) != null && (n = tab.length) > 0 &&
            (first = tab[(n - 1) & hash]) != null) {
    ...
} 
```

首先判断 table 是否为空，如果不为空，则将索引位置上的元素赋予临时变量 first，并在验证为非 null 值后

```java
if (first.hash == hash && // always check first node
                ((k = first.key) == key || (key != null && key.equals(k))))
    return first;
```

首先检查 first 节点是否是我们要找的节点，找到的话直接返回。否则

```java
if ((e = first.next) != null) {
    if (first instanceof TreeNode)
        return ((TreeNode<K,V>)first).getTreeNode(hash, key);
    do {
        if (e.hash == hash &&
            ((k = e.key) == key || (key != null && key.equals(k))))
            return e;
    } while ((e = e.next) != null);
}
```

如果 first 是红黑树的节点，则在红黑树中查找，否则从链表中查找。可以发现 get 方法的内容比 put 方法简单得多，可谓出去容易进去难。。这次的分析就先到这里了。

