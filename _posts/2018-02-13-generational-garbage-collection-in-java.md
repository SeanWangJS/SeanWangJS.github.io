---
layout: post
title: Java 中的分代垃圾收集思想
tags: 垃圾回收器 Java虚拟机
---

程序员总是喜欢把繁琐的事情交给机器去自动执行，特别是具有固定模式的任务，大家都不愿意亲历亲为。而对于那些一眼看上去不存在什么通用特征的事情，就会努力去在更高的层次上去寻找它们的泛性。所以对于初学者来说，很多知识看起来抽象得不行，就是因为还没有从底层开始逐渐构造这些概念。

程序内存资源的管理逻辑很简单，一句话就是：不用的资源请释放出来。但是却很繁琐，对于程序员来讲，那么多代码，他怎么记得住自己在什么时候申请了多少内存？哪些资源以后还可能要用？哪些内存根本就不必回收？如果自己来手动管理内存，不知要浪费多少脑细胞，所以垃圾回收器这时候就帮了大忙了。

但是自动垃圾回收不可能像人的判断那样准确，所以它的宽容度其实是要高很多的。另一个特征是自动垃圾回收过程其实是独立于程序执行的，它不像把释放资源命令直接写在代码里，在程序运行过程中就顺便释放了，而是在垃圾回收时，停下所有任务，把该回收的内存释放掉，该搬走的对象搬走，然后再恢复原先的线程。
分代垃圾回收这一套方法其实基于两个假设：

1.	大部分新资源很快会变成垃圾，需要回收内存。
2.	如果有资源在很长的时间里都没有被回收，那它就不太可能被回收了。

注意，这只是两条假设，并不是什么定理，更不是公理，所以我们看到了一些不确定的字眼，这并不影响。事实证明，大部分的程序都符合这两条假设，这才使得现行的一些回收算法成为可能。

初略地讲，分代垃圾回收要求把内存分成两个主要部分，即新生代和老生代。顾名思义，新的对象会被分配到新生代内存区，而老生代则存着旧对象。当触发垃圾回收的时候，程序会扫描新生代区域，发现无用对象，进行内存回收。如果经过几次新生代垃圾回收后，发现有的对象始终没有被回收，那么根据假设，这个对象以后也不太会被回收了，于是就将其整个搬到老生代空间。如果老生代空间都满了，那就触发对老生代的垃圾回收。可以看到，这里将回收行为分成了两个层次，一个是在新生代中的垃圾回收，一个是在老生代中，可以看到，在新生代中的垃圾回收频率是要高于老生代中的。由于垃圾回收会使程序暂停，并且扫描空间越大，暂停时间也越长，如果控制新生代的空间大小，就能显著减小其中的垃圾回收时间。所以分代垃圾收集的好处在这里就显现了，它能对内存空间进行更细粒度的管理，也就能实现更高效的算法。

而新生代内存区，又进一步被划分成Eden空间和Survive空间。因为我们看到，在扫描新生代的时候会检查旧对象是否可以被移动到老生代去，于是可以把旧对象统一到Survive空间进行管理，而新对象则总是分配到Eden空间，这样每次检查只需要扫描Survive空间就行了，这就进一步减小了时间开销。

*小结*

Java的垃圾回收机制还是比较复杂的，本文只从理念的角度来表述了下分代垃圾回收的思想框架，还有更多的细节内容需要考虑，留在以后再来说明。

——堵在高速路上作