---
title: 代码片段：全排列算法
tags: 代码片段
---

给定无重复元素的集合 Set，大小为 n，计算这 n 个值所有可能的排列方式。

首先我们给出一个构建正确排列的方法：先随机选择第一个元素，然后在剩余的元素中选择第二个元素，再从剩余的元素中选择第三个元素，如此下去，直到确定了最后一个元素。

多次使用上述方法便能获得 Set 的多个排列，但是为了解决问题，我们需要先回答三个问题

1. 在构造排列的过程中，如何保证不重复选取元素
2. 如何保证构造的所有排列都是不重复的
3. 如何保证我们构造的排列集合覆盖了 Set 的所有排列方式

下面我们逐一分析，令集合 Set 在程序中以数组表示为

```c
set = [a b c d e]
```

这是有5个不同元素的数组，假如我们第一次选择了第 1 个元素 a，为了不重复，接下来需要从第 2 到第 5 个元素中选择



