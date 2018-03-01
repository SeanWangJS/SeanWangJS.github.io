---
layout: post
title: 反向传播算法
tags: 神经网络
modify_date: 2018-03-01
---

考虑下面这样一个三层网络结构

![](/resources/2018-02-28-back-propagation-algorithm/network.png)

其中输入特征的的维度为 m，输出数量为 p，中间隐含层数量为 n，并且每一层的输出值都标识在了节点上。我们首先从输出层开始分析，对于节点 $$y_k$$ ，假设它与隐含层各节点的权重系数为 $$w_{jk},j=1,2,,,m$$。如下图所示

![](/resources/2018-02-28-back-propagation-algorithm/output.png)

然后再假设激活函数为 $$f$$ ，于是就有

$$
s_k = \sum_{j=1}^m w_{jk} h_j\\
y_k=f(s_k)
$$

其中用 $$s_k$$ 来存储中间的求和结果。如果我们定义样本的真实输出值为序列 $$\hat y_k$$ ，那么可以得到误差函数

$$
E= \frac 1 2\sum_{k=1}^p (y_k - \hat y_k)^2
$$

接下来我们再考虑隐含层单元 $$h_j$$ ，假设它与输入层各节点的权重系数分别为 $$w_{ij}',i=1,2,,,n$$ 。如下图所示

![](/resources/2018-02-28-back-propagation-algorithm/hidden_layer.png)

经过计算后得到

$$
s_j' = \sum_{i=1}^m w_{ij}' x_i\\
h_j = f(s_j')
$$

上式中我们用 $$s_j'$$ 来存储中间的求和结果。由于神经网络的训练其实就是不断调整权重参数使得最终输出符合给定结果的过程，如果要利用梯度信息对参数进行更新，则需要求解误差函数对于权重的导数，即

$$
\frac {\partial E} {\partial w_{ij}},\frac{\partial E}{\partial w_{ij}'}
$$

我们先考虑第一个导数的计算，由于 $$E$$ 是 $$y_k$$ 的函数， $$y_k$$ 是 $$s_k$$ 的函数， $$s_k$$ 又是 $$w_{ij}$$ 的函数，所以利用链式求导关系，可得

$$
\frac{\partial E}{\partial w_{jk}} = \frac{\partial E}{\partial y_k}\frac{\partial y_k}{\partial s_k}\frac{\partial s_k}{\partial w_{jk}} = (y_k-\hat y_k)f'(s_k)h_j
$$

接下来考虑第二个导数 $$\frac{\partial E}{\partial w_{ij}'}$$ 的求解，由于 $$s_j'$$ 是 $$w_{ij}'$$ 的函数，所以

$$
\frac{\partial E}{\partial w_{ij}'} = \frac{\partial E}{\partial s_j'}\frac{\partial s_j'}{\partial w_{ij}'} =  \frac{\partial E}{\partial s_j'}x_i
$$

而从网络的传播方式来看，$$s_j'$$ 会影响后续的所有 $$s_k$$ ，于是有

$$
 \frac{\partial E}{\partial s_j'} =  \sum_{k=1}^p\frac{\partial E}{\partial s_k} \frac{\partial s_k}{\partial s_j'}
$$

这一结论仍然来自于链式求导法则，它的表述如下：

>已知关于 $$x$$ 的函数 $$f(x)$$ ，如果 m，n也是关于 $$x$$ 的函数，那么就有 $$\frac{\partial f}{\partial x} = \frac{\partial f}{\partial m}\frac{\partial m}{\partial x} + \frac{\partial f}{\partial n} \frac{\partial n}{\partial y}$$。

我们继续推导

$$
\begin{aligned}
 \frac{\partial E}{\partial s_j'} &=  \sum_{k=1}^p\frac{\partial E}{\partial s_k} \frac{\partial s_k}{\partial s_j'}\\
 &=\sum_{k=1}^p \frac{\partial E}{\partial y_k}\frac{\partial y_k}{\partial s_k} \cdot \frac{\partial s_k}{\partial h_j}\frac{\partial h_j}{\partial s_j'}\\
 &=\sum_{k=1}^p (y_k-\hat y_k)f'(s_k) w_{jk} f'(s_j')
 \end{aligned}
$$

所以，综合起来有

$$
\frac{\partial E}{\partial w_{ij}'}=\sum_{k=1}^p (y_k-\hat y_k)f'(s_k) w_{jk} f'(s_j') x_i
$$

再结合前面推导出的

$$
\frac{\partial E}{\partial w_{jk}}  = (y_k-\hat y_k)f'(s_k)h_j
$$

如果定义

$$
\delta_k = (y_k-\hat y_k)f'(s_k)\\
\delta_j' = \sum_{k=1}^p \delta_k w_{jk}f'(s_j')
$$

那么上面两个导数可分别化简为

$$
\frac{\partial E}{\partial w_{ij}'} = \delta_j'x_i\\
\frac{\partial E}{\partial w_{jk}} = \delta_k h_j
$$

上式的形式十分简洁，并且我们可以从 $$\delta_j$$ 的定义发现一丝递推关系的意味。实际上这里确实存在递推关系，下面我们来考虑多层网络，为了方便书写递推式，我们重新规定一下符号。

![](/resources/2018-02-28-back-propagation-algorithm/mul_layer.png)

我们仍然用 $$(x_i)_ 0^n$$ 表示初始输入，$$(y_i)_ 0^p$$ 表示最终输出，但为了统一描述，又分别给它们加上了别名在旁边，即 $$(h_i^0)_ 0^{m_0} $$ 和 $$(h_i^{t+1})_ 0^{m_{t+1}} $$ 。隐含层用 $$h$$ 表示，共有 $$t$$ 层，层号用 $$h$$ 的上标表示，每层的节点序号用下标表示。另外权重用 $$(w^i)_ 1^{t+1}$$ 表示，其中的每个元素 $$w^i$$ 都是一个矩阵，并且 $$w_{jk}^i$$ 表示第 i-1 层的第 j 个节点与第 i 层的第 k 个节点间的权重。最后用 $$s_j^i$$ 表示第 i 层第 j 个节点的加权和

$$
s_j^i=  \sum_{k=0}^{m_{i-1}}w_{jk}^ih_k^{i-1}\\
h_j^i = f(s_j^i)
$$

运用上述符号，以及结合前面的推导，我们可以得到误差函数 $$E$$ 关于最后两层间权重系数 $$w_{jk}^{t+1}$$ 的导数

$$
\frac{\partial E}{\partial w_{jk}^{t+1}}  = (y_k-\hat y_k)f'(s_k^{t+1})h_j^t
$$

以及重新定义符号

$$
\delta_k^{t+1} = (y_k-\hat y_k)f'(s_k^{t+1})\\
\delta_j^t= \sum_{k=1}^{m_{t+1}} \delta_k^{t+1} w_{jk}^{t+1}f(s_j^t)
$$

于是有

$$
\frac{\partial E}{\partial w_{ij}^t} = \delta_j^th_i^{t-1}\\

\frac{\partial E}{\partial w_{jk}^{t+1}} = \delta_k^{t+1} h_j^t
$$

当然我们此时可以不仅仅局限于倒数第一层和倒数第二层，对于任意层都可以得到通用公式

$$
\delta_j^i =  \sum_{k=1}^{m_{i+1}} \delta_k^{i+1} w_{jk}^{i+1}f(s_j^i)\\
\frac{\partial E}{\partial w_{jk}^i} = \delta_k^i h_j^t
$$

可以看到，上述公式中，我们通过初始的 t+1 层 $$\delta_k^{t+1}$$ 逐层向前，计算误差对前层权重的导数，这就是反向传播算法之所以叫这个名称的原因所在。在得到误差对各层权重参数的导数之后，就能利用梯度信息对权重进行修正，比如梯度下降算法

$$
\bar {w_{jk}^i} = w_{jk}^i - \alpha \frac{\partial E}{\partial w_{jk}^i} = w_{jk}^i - \alpha \delta_k^i h_j^i
$$

很抱歉我的符号已经找不到地方放了:)，这里就以 $$\bar {w_{jk}^i}$$ 表示更新后的值。













