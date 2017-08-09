---
layout: default
---

## Java HashMap源码分析(1)

哈希表是一种常用而强大的数据结构，有鉴于此，许多语言都自带其实现，这就为我们这些使用者带来了很多方便。在Java中，HashMap是使用率最高的哈希表，本文将对其源码进行详细的解读。
HashMap是一个比较大的类，有两千多行代码，继承了一个类，实现了三个接口，以及大量的方法。如果上来就一行一行的读，那么将很快就陷入泥潭，无法自拔。所以我决定在看源码的同时，写一个简化版本的哈希表，一开始只关注哈希表本身，然后再逐步添加实际应用层面的功能（比如，我将先不考虑HashMap中的最大容量、序列化等等与哈希表结构无关的内容）。然后我想说下，我发现在IDEA中类名使用小写首字母并不会弹出警告，所以我决定在不造成困扰的情况下不遵从Java常用的驼峰命名规则，比如接下来我要实现的简化哈希表就叫hashmap。

### 哈希表索引下标计算

我们都知道数组这种结构，只要知道索引号，就能经过一步操作获得对应的值，这里的索引号以及对应的值便成为了一个键值对，但这里的键必须是正整数。整数类型的键很明显存在局限性，比如有个需求要通过人名找到对应的人物信息，如果想要常数时间复杂度，就不存在对应的数组解决方法。于是哈希表便在类似的需求下应孕而生，其本质上是一个数组（**在HashMap中用Node<K, V>[] table 表示**），只不过非整数类型的键被哈希函数转换成整数，即得到索引。知道了这一点，那问题就很清晰了，下面给出了最简单的哈希表结构伪码。


```java
public class hashmap<K, V>{

    private node<K, V>[] table;
    
    public V get(K key) {
         int index = hash(key);
         return table[index].value;
    }
    
    public void put(K key, V value) {
         int index = hash(key);
         table[index] = new node(key, value);
    }
    static class node<K, V> {
         K key;
         V value;
         node(K key, V value) {
             this.key = key;
             this.value = value;
         }
    }
}
```

hashmap用内部类 node 来管理键值对，然后使用 node 数组 table 作为键值对容器，之所以用“节点”这个词，是因为在必要时要将这个块扩展为链表或者树，这将在后面讨论。这里的hash函数，接受key，然后返回对应的索引，并且不同的key对应不同的索引，否则数据就被后来者覆盖了，一种很自然的想法是利用Java对象自带的哈希码方法，即hashCode()，来实现这一功能。但问题在于hashCode()计算的值实在太大，总是开辟这样大的数组空间显得不现实。于是HashMap的设计者就想到一个简单粗暴的方法，使用不那么大的数组容量n，然后将哈希码与n - 1作&运算。
```java
h = key.hashCode();
index = h & (n - 1);
```

这样得到的index肯定比n小。但这种方式有个问题是，只利用了哈希码的低位，如果n比较小，那么出现碰撞的可能性就会很大。现在来看看HashMap中是如何处理的，其hash函数如下

![](/resources/2017-08-01-hashmap-source-code-01/hash-func.png)

也就是说，这里首先计算key的哈希码，然后在让其高16位和低16位作异或运算，下图以h = 989051363 为例，叙述了这一过程

![](/resources/2017-08-01-hashmap-source-code-01/hash-calc.png)

经过这样的处理，再和(n - 1)作&运算，让哈希码的高位也参与到了索引计算，设计者认为有助于减少碰撞。为了检验这种操作是否能达到理想的效果，可以做个实验看看。首先生成一万个介于0～10000000的正整数来模拟哈希码，然后再选取合适的n，使用两种方式来计算索引，即 h & (n - 1) 和 (h ^ (h >>> 16)) & (n - 1)，最后比较两种方式造成的碰撞结果

![](/resources/2017-08-01-hashmap-source-code-01/compare.png)

上面第一列表示容量n，第二列m1和第三列m2分别是，异或之后计算的索引和直接计算的索引中，没有出现碰撞的数量。可以发现，异或过程确实能够减少碰撞，虽然这里的效果不是太明显。在源码中hash方法上的注释写道：

>There is a tradeoff between speed, utility, and quality of bit-spreading. Because many common sets of hashes are already reasonably distributed (so don't benefit from spreading), and because we use trees to handle large sets of collisions in bins, we just XOR some shifted bits in the cheapest possible way to reduce systematic lossage, as well as to incorporate impact of the highest bits that would otherwise never be used in index calculations because of table bounds.

因为现有的哈希分布已经很理想了，再修正哈希码的收效并不大，所以，异或运算是一个权衡了速度、效用和质量后的决定。而对于那些已经发生碰撞的数据，则使用树来存储。

另外，为了不重复调用 hash 函数，可以在 node 类中声明一个字段用于存储相应 key 的 hash 值，将 node 类修改如下
```java
static class node<K, V> {
	final int hash;
    final K key;
    V value;
    node(int hash, K key, V value) {
		this.hash = hash;
        this.key = key;
        this.value = value;
    }
}
```
将 key 和 hash 字段声明为 final 是为了保持这些量的不变性。

### 存入数据

从上面的测试可以发现，即便数组容量n比元素数量大，还是会出现很多哈希碰撞，于是在HashMap中就定义了loadFactor（负载因子）的概念。当存入的数据量 size 增加到一定程度（**由 threshold 界定**）的时候，就对数组进行扩容（扩容过程将在下面讨论），比如说，loadFactor = 0.75，n = 16384，那么当 size = loadFactor * n = 12288 时，便重新计算数组容量，并拷贝数据到新数组。有了这种设计，HashMap的构造方法其实就是在initCapacity（初始容量）和loadFactor（负载因子）之间折腾。

![](/resources/2017-08-01-hashmap-source-code-01/constructor.png)

第一个构造方法同时指定了初始容量和负载因子，第二个方法只指定了初始容量，而负载因子使用默认值，第三个方法无参数，负载因子取默认值，没有指定初始容量。第四个方法用另一个哈希表来初始化，暂先不考虑。这里最重要的方法就是第一个构造函数，如下

![](/resources/2017-08-01-hashmap-source-code-01/constructor-i-f.png)

前面几行在进行参数检查，值得注意的是最后一行的tableSizeFor方法:

![](/resources/2017-08-01-hashmap-source-code-01/table-size-for.png)

这个函数接受容量作为参数，然后返回一个是2的幂的数，并赋值给threshold变量（threshold是数组扩容标志，如果 size > threshold，则进行扩容操作）。也就是说，如果我在初始化HashMap时指定了初始容量为90，实际上它初始化的数组大小并不是90，而是比90大的那个为2的幂的数128。这里的实现非常不直观，但是一看就觉得应该很高效。。。实际上面过程只做了一件事，就是把 cap 的二进制最高位的那个1后面的位全部置为1，返回加1后的结果。比如说 cap 的二进制表示为 1011010 ，那么这个函数其实就是返回了 1111111 + 1，这就肯定是2的幂。函数里面的一系列位运算其实就是在实现这一目标，当cap =  1011010 时，n = 1011001，那么 n >>> 1 的第六位肯定为1，再和n作\|运算，得到的n的第6、7位肯定都为1。第二步 n \|= n >>> 2 之后，n的第4、5、6、7位肯定都为1。继续进行这一过程，最终得到1～7位都为1的结果，再加1返回，便达成目标。

之所以要进行5步 \| 运算，是因为Java中int类型有32位，这样即便一个整数的第31位是1，也能保证返回正确的结果。而函数第一行要对 cap 减1，否则，如果 cap 本身就是2的幂，那么得到的将是 2×cap ，这就过度开辟了存储空间。

现在回到构造函数，发现在这里并没有实际初始化容器数组，而有的构造函数甚至都没有指定初始容量。这些工作还需要推迟到 put 方法中去，但我现在还不能理解为何这些构造方法会如此蹩脚，所以先写一个简化版本的构造函数集

```java
public hashmap(int initialCapacity, float loadFactor) {
        if(initialCapacity < 0) {
            throw new IllegalArgumentException("Illegal initial capacity: " + initialCapacity);
        }
        if(loadFactor <= 0 || Float.isNaN(loadFactor)) {
            throw new IllegalArgumentException("Illegal load factor: " + loadFactor);
        }
        this.loadFactor = loadFactor;
        table = new node[tableSizeFor(initialCapacity)];
    }

    public hashmap(float loadFactor) {
        this(DEFAULT_INITIAL_CAPACITY, loadFactor);
    }

    public hashmap() {
        this(DEFAULT_INITIAL_CAPACITY, DEFAULT_LOAD_FACTOR);
    }
```
在HashMap源码中，put函数调用了 putVal 方法

![](/resources/2017-08-01-hashmap-source-code-01/put-val.png)

putVal 方法接受前面计算的 hash 值，以及一组键值对，首先判断容器数组是否为空，如果是，则先调用 resize 方法初始化容器数组。如果不为空，则判断对应索引上是否有值，如果没有，则将数据放到该位置，如果有，则使用链表或者二叉树来存储相同索引数据，现在先不管这种情况。这里值得注意的是 resize 方法

![](/resources/2017-08-01-hashmap-source-code-01/resize.png)

为了简便起见，所有涉及到最大容量检查的代码都忽略不看。下面分情况讨论

1. 方法内首先检查旧容器数组的大小是否为0，如果大于0，则新的数组容量翻倍，并且如果旧数组的容量大于等于默认值，则新的threshold 也翻倍。这种情况对应于HashMap对象已经使用了put方法的情况，因为一开始数组容量一定是等于0的（当然这是在只考虑前三个构造方法的假定下）。
2. 如果旧数组容量等于0，但是 threshold 大于0，那么就将threshold的值作为新的数组容量，这种情况对应于调用 HashMap(int initialCapacity, int loadFactor) 或者 HashMap(int initialCapacity) 新建对象，但是还没有进行 put 操作的情况，这时的 threshold 一定是2的幂，所以可以直接作为数组容量。
3. 如果数组容量和 threshold 都等于0，则用默认值作为数组容量，threshold 也用默认负载因子初始化。这种情况对应于调用 HashMap() 新建对象，但是还没进行 put 操作的情况。

接下来又判断了新的 threshold 是否等于0，如果等于0，那么就用正常的程序走一遍。最后用新的容量大小新建容器数组，接下来开始进行数据的拷贝操作，也就是说，将原来的数据复制到新数组中新的位置，这个位置仍然由数据key的 hash 以及数组容量n决定，即 index = hash & (n - 1)。假设扩容前的容量为 n1，由于n1 和 n 都是2的幂，则 n1 - 1 和 n - 1的二进制表示，实际上只相差一个最高位上的1，比如 n1 - 1= 00001111 ，n - 1 = 00011111。在这种情况下，如果hash的第5位（我称作标志位）为 0，假设 

hash = 0000abcd

那么

indexOld  = hash & (n1 - 1) = 0000abcd & 00001111 = 0000a'b'c'd'

indexNew = hash & (n - 1)   = 0000abcd & 00011111 = 0000a'b'c'd'

也就是说 indexOld = indexNew ，而如果标志位为 1，即 hash = 0001abcd ， 那么

indexOld  = hash & (n1 - 1) = 0001abcd & 00001111 = 0000a'b'c'd'

indexNew = hash & (n - 1)   = 0001abcd & 00011111 = 0001a'b'c'd'

而 n1 恰好等于 00010000，于是 indexNew = n1 + indexOld，所以通过上面两种情况的分析，在不重新计算 hash 的情况下确定数据索引，这是一种相当精妙的设定。基于以上描述，可以实现简化的 resize 方法

```java
public void resize() {
	int capacity = table.length;
    int newCap;
    if(capacity > 0) {
        newCap = capacity << 1;
    }
    else {
        newCap = DEFAULT_INITIAL_CAPACITY;
    }

    node<K, V>[] newTab = (node<K, V>[])new node[newCap];

    for(int i = 0; i < capacity; i++) {
        node<K, V> e = table[i];
        if(e != null) {
        //判断hash的标志位是否为0
            if((e.hash & capacity) == 0) {
                newTab[i] = e;
            }else{
                newTab[i + capacity] = e;
            }
        }
    }

    table = newTab;
}	
```

根据前面对 putVal 的分析，可以写出简化的 put 方法
```java
public hashmap put(K key, V value) {
	if(table.length == 0 || size >= (int) (table.length * loadFactor)) {
        resize();
    }
    int hash = hash(key);
    table[hash & (table.length - 1)] = new node<>(hash, key, value);
    size++;
    return this;
}
```

注意上面没有对两个数据有相同hash值的情况进行处理，而是直接覆盖，以后将会讨论这个问题。

### 查询数据

在哈希表中查询数据操作的时间复杂度为O(1)，与 put 过程类似，首先计算 key 的索引，代码如下
```java
public V get(K key) {
    int index = hash(key) & (table.length - 1);
    return table[index].value;
}
```

### 遇到相同 hash 情况

如果向哈希表中put键值对的时候，发现插入位置还有其他的数据，那么就需要进一步处理，先看看 HashMap 类的 putVal 函数是怎么做的

![](/resources/2017-08-01-hashmap-source-code-01/put-val2.png)

上面代码的意思是：

1. 如果新插入数据的 hash 值等于已经存在的数据 p 的 hash 值，并且两者的 key 相同，这种情况其实是在更新键值，可以直接覆盖旧数据。
2. 如果已有数据 p 是一个树节点，那么新数据就需要插入到树里面。
3. 最后一种情况意味着 p 是一个链表节点，如果发现新数据的 key 其实就是链表中某个节点的 key，则用新的value覆盖，如果一直没发现同 key 数据，则需要将新数据插入到链表末尾。binCount 的终值记录了链表长度，如果发现链表长度超过了阈值 TREEIFY_THERSHOLD ，那么就把链表转换成二叉树（确切的说是一棵红黑树）。

为了减少复杂性，现在先不考虑二叉树的存储结构，当发现索引位已有数据时，直接用链表来存储新数据。当然在此之前要先为 node 类添加一个 next 字段，用于存储后续节点引用。

```java
static class node<K, V>{

	//code
	
	node<K, V> next;	
	
	//code
}

```

然后是依照上述规则更新后的 put 方法，由于不用考虑将链表转换成树，所以在遍历链表时没有记录其大小。

```java
public hashmap put(K key, V value) {
        if(table.length == 0 || size >= (int) (table.length * loadFactor)) {
            resize();
        }
        int hash = hash(key);
        int index = hash & (table.length - 1);
        node<K, V> p = table[index];
        K k;
        if(p != null) {
            //如果新旧数据的键相同，则只需更新 value
            if(p.hash == hash && ((k = p.key) == key || (key != null && key.equals(k)))) {
                p.value = value;
            }else {
                node<K, V> e;
                //遍历节点p开头的链表，如果在链表中发现key，则对value进行更新。否则将新的键值对插入链表末尾
                while(true){
                    e = p.next;
                    if(e == null) {
                        p.next = new node<>(hash, key, value);
                        size++;
                        break;
                    }
                    if(e.hash == hash && ((k = e.key) == key || key != null && key.equals(k))) {
                        e.value = value;
                        break;
                    }
                    p = e;
                }
            }
        }else{
            table[index] = new node<>(hash, key, value);
            size++;
        }
		if(size >= table.length * loadFactor) {
            resize();
        }
        return this;
    }
```

### 小结

以上，一个具有基本功能的哈希表被建立起来了，大体上分析了HashMap的构造函数，tableSizeFor方法，put方法，resize方法以及 get 方法。这只是HashMap中非常基础的部分，当然也是核心的内容，其他的代码基本上就是在维护这种结构，或者是提供一些友好的接口，方便开发人员的操作。

**引用声明**：本文主要参考了YiKun的 [Java HashMap 工作原理及实现](http://yikun.github.io/2015/04/01/Java-HashMap%E5%B7%A5%E4%BD%9C%E5%8E%9F%E7%90%86%E5%8F%8A%E5%AE%9E%E7%8E%B0/)，以及Java HashMap类源码。
