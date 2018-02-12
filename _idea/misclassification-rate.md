---
layout: post
title: 模式识别中的最小错误率决策
tags: 模式识别系列
---

对于一个有监督的分类学习问题，如果我们把注意力仅集中在分类的对错上面，就自然会想到使用一个函数来度量分类的错误率，并且在学习的过程中尽量减小训练数据的分类错误率，这就是基于最小错误率的决策方法。

假设我们拿到了一组特征数据及其类别标签

$$
S = \{(x_i, y_i)\mid i\in N^+, i\le n ,\, \,y_i \in \{\mathcal{C}_j\},\,j=1,2,,,k\}
$$

也就是说，总共有 *n* 个特征，它们分属于 *k* 个类别。如果特征的维度是 *d* ，那么可以认为特征空间是一个 *d* 维实向量空间 $$R^d$$ 的子集。分类行为其实就是在计算一个从特征空间到类别集合的映射，我们定义该映射如下

$$
f:R^d \rightarrow C
$$

这里的 $$C = {\mathcal{C}_i}\, , i = 1,2,,,k$$ 。由于 *d* 维实向量空间本身就是一个群（既是加法群，又是乘法群）。所以我们可以在其中定义等价关系，即：

如果 $$x_i$$ 和 $$x_j$$ 都被映射到同一个类别：$$f(x_i)=f(x_j)$$ ，那么两者等价。

可以看到，这个关系满足自反性，对称性以及传递性，所以这确实是一个等价关系。于是特征空间可以通过这一等价关系被划分成 *n* 个等价类，定义为 $$R_i\,,\,i = 1,2,,,n$$，它们都是特征空间的子集，但互不相交，且有

$$
f(x \in R_i) = \mathcal{C}_i
$$

对于一个特征 $$x$$，我们对它进行分类的过程其实就是计算 $$f(x)$$ ，看结果属于哪一类别，于是 $$f(x)$$ 就是我们的预测结果，它和特征的实际类别属性不是一个概念，但却相关，定义两者的联合分布为

$$
p(f(x), \mathcal{C}_j)
$$

其中 $$f(x)=\mathcal{C}_i$$ ，这里的 *i* 是与 *j* 独立的量。显然，要使 $$f(x)=\mathcal{C}_i$$ ，则必然有 $$x \in R_i$$，反之亦然，所以上面定义的联合分布又等价于

$$
p(x\in R_i, \mathcal{C}_j)
$$

如果分类正确，那么就必须使得 $$i=j$$ ，所以将一个特征分类到 $\mathcal{C}_i$$ 并且还正确的概率就为

$$
p(x\in R_i, \mathcal{C}_i) = \int_{x\in R_i} p(x, \mathcal{C}_i)\mathbf{d}x
$$

将二元组 $(x\in R_i, \mathcal{C}_i)$$ 作为一个事件，那么我们可以定义正确分类的事件为如下形式

$$
\bigcup_{i=1}^n (x\in R_i, \mathcal{C}_i)
$$

由于上述事件的互斥性，于是正确分类的概率就为

$$
\begin{aligned}
p(correct) &= p\left(\bigcup_{i=1}^n (x\in R_i, \mathcal{C}_i)\right) \\
&= \sum_{i=1}^n p(x\in R_i, \mathcal{C}_i) \\
&= \sum_{i=1}^n \int_{x\in R_i} p(x, \mathcal{C}_i)\mathbf{d}x
\end{aligned}
$$

可以看到，当 $$x \in R_j$$ 的时候，分类正确的概率就为

分类是模式识别中的基本问题，它通过分析数据的特征决定应被归入的类别。实际应用中的分类模型多种多样，但是它们的原理基本都是通过对特征进行聚类来判断其所属类别。比如将特征空间分为 *n* 个区域 $$R_1,R_2,,,R_n$$ ，并且这些区域互不重叠，也就是 $$R_i \cap R_j = \emptyset$$，对任意 $$i\ne j$$ 都成立。那么对于一个特征 *x* ， 只需要观察它所在的区域就能判断其所表示的类别，这里我们用 $$\mathcal{C}_1, \mathcal{C}_2,,,\mathcal{C}_n$$ 来表示种类。

一次分类行为应该区分清楚两个事件，其一是通过一定的方式计算出特征所在的区域，其二是特征本身所处的类别。第一个事件显然是我们做出判别的依据，它可以表示成 $$x \in R_i $$，而第二个则用类别的标识来表示即可 $$\mathbf{C}_j$$。显然，这两个事件都是随机的，我们可以将它们的概率表示出来，即

$$
p(x\in R_i)\,, \quad p(\mathcal{C}_j)
$$

其中 $$p(\mathcal{C}_j)$$ 是在不知道任何有关特征的信息下，类别 $$\mathcal{C}_j$$ 出现的概率，也被称为先验概率。它们的联合分布为

$$
p(x \in R_i, \mathcal{C}_j)
$$

如果在上述联合分布中，$$i = j$$，即 $$x\in R_i$$ 和 $$x$$ 本身属于类别 $$\mathcal{C}_i$$ ，也就是两个事件同时发生，就说明分类是正确的，否则就是错误的。对于一般特征，若不具有任何先验知识，它被正确分类的概率可以定义为

$$
p(correct) = \sum_{i=1}^n p(x\in R_i, \, \mathcal{C}_i)
$$

至于为什么这个量可以作为概率值，我们可以作如下考虑

$$
\begin{aligned}
p(correct) &= \sum_{i=1}^n p(x\in R_i, \, \mathcal{C}_i)\\
&=\sum_{i=1}^n \int_{x\in R_i} p(x, \mathcal{C}_i)\mathbf{d}x\\
&\le \sum_{i=1}^n \int_{-\infty} ^{\infty}p(x, \mathcal{C}_i) \mathbf{d}x\\
&=\sum_{i=1}^n p(\mathcal{C}_i)\mathbf{d}x\\
&= 1
\end{aligned}
$$

也就是说，$$p(corrent)$$ 始终小于 1，符合概率密度函数的要求。根据概率的互斥关系，它被错误分类的概率等于 1 减去正确率

$$
p(error) = 1 -p(correct)
$$

到这里，我们的任务就已经很明显了，那就是想尽办法提高分类正确率，或者说减小错误率，显然在这一层面上，最好的决策使错误率最小，这就是基于最小错误率的决策方案。

这里我们需要思考一下，对 $$p(correct)$$ 求最大值，其中的变量是什么？由于

$$
p(correct) = \sum_{i=1}^n \int_{x\in R_i}p(x,\mathcal{C}_i)\mathbf{d}x
$$



为了计算正确率，考虑如下推导

$$
\begin{aligned}
p(correct) &= \sum_{i=1}^n p(x\in R_i, \, \mathcal{C}_i)\\
&=\sum_{i=1}^n \int_{x\in R_i}p(x, \mathcal{C}_i)\mathbf{d}x\\
&=\sum_{i=1}^n \int_{x\in R_i}p(\mathcal{C}_i\mid x)p(x)\mathbf{d}x
\end{aligned}
$$














end

end
