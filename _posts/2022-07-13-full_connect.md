---
title: 全连接层的前馈计算与反向传播分析
tags: 神经网络
---

全连接层差不多可以说是神经网络中最简单的结构，对它进行分析可以让我们比较容易地建立起对神经网络运作方式的理解基础。首先我们来看一下前馈计算过程，设置全连接层的输入维度为 \\(n\\)，输出维度为 \\(m\\)，并定义以下变量：输入向量 \\(x \in R^m\\)，输出向量 \\(y\in R^n\\)，权重向量 \\(W \in R^{n\times m}\\)，偏置项 \\(b \in R^n\\)，则前馈计算的公式表示如下

$$
  y = W x + b\qquad \qquad (1)
  $$

它的分量形式为

$$
  y_i = \sum_{j = 1}^m w_{ij}x_j + b_i \qquad \qquad (2)
  $$

前馈计算相当简单，接下来我们再考虑它的反向传播，也就是损失对参数 \\(W, b\\) 求梯度的过程，假设损失函数为 \\(L\\)，这是一个跟 \\(y\\) 相关的函数，根据复合函数求导的规则，有以下公式

$$
  \frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial W}  \qquad \qquad (3)
  $$

由于向量对矩阵的导数是一个三阶张量，为了方便起见，我们使用张量形式来表示上面各项

$$
  \begin{aligned}
  \frac{\partial L} {\partial W} &= r_{ij}e_i e_j, \quad &r_{ij} = \frac{\partial L}{\partial w_{ij}}\\
  \frac{\partial L} {\partial y} &= p_i e_i, &\quad p_i = \frac{\partial L}{\partial y_i}\\
  \frac{\partial y} {\partial W} &= q_{kij} e_ke_i e_j , \quad& q_{kij} = \frac{\partial y_k}{\partial w_{ij}}
  \end{aligned} 
  \qquad \qquad (4)
  $$

于是在张量形式下，公式 (3) 可以写为

$$
  \begin{aligned}
  &r_{ij} e_i e_j = p_k e_k \cdot q_{kij} e_k e_i e_j = p_k q_{kij} e_i e_j\\
  \Rightarrow &r_{ij} = p_k q_{kij}
  \end{aligned} 
  \qquad \qquad (5)
  $$

再根据 \\(y_i\\) 的具体计算公式(2)，我们可以求得 \\(q_{kij}\\) 等于

$$
  q_{kij} = \frac{\partial y_k}{\partial w_{ij}} = \left\{\begin{aligned}
x_j &\quad \mathrm{if} & i = k \\
0 &\quad \mathrm{if} & i \neq k
  \end{aligned}\right.
  \qquad \qquad (6)
  $$

然后把它带入公式(5)，可以得到

$$
  r_{ij} = p_i x_j \qquad \qquad (7)
  $$

最后还原成矩阵形式，我们就有 

$$
  \frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \otimes x
  \qquad \qquad (8)
  $$

其中 \\(\otimes\\) 是向量间的张量积符号。

对于偏置项来说，同样利用复合函数求导的规则

$$
  \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial b}
  $$

写成张量形式

$$
  \frac{\partial L}{\partial b_j} e_j= \frac{\partial L}{\partial y_{i}} e_i \cdot \frac{\partial y_i}{\partial b_j} e_i e_j
  $$

再考虑到公式 (2)，可得 \\(y_i\\) 对 \\(b_j\\) 的偏导数为

$$
  \frac{\partial y_i}{\partial b_j} = \left\{
    \begin{aligned}
    &1 \quad \mathrm{if} \quad i = j\\
    &0 \quad \mathrm{otherwise}
    \end{aligned}
    \right.
  $$

所以 

$$
  \frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial y_{i}}
  $$

还原到矩阵形式

$$
  \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y}
  $$

以上就是全连接层在单个输入情况下的前馈和反向传播计算过程，考虑到神经网络的前馈计算一般都是批量输入的，于是还有必要将反向传播过程扩展到批量输入的形式。假设 batch_size 等于 \\(d\\)，定义输入张量为 \\(X \in R^{m \times d}\\)，输出张量为 \\(Y \in R^{n\times d}\\)，偏置项为 \\(B \in R^{n \times d}\\)，权重系数矩阵仍然为 \\(W \in R^{n\times m}\\)。这时前馈计算公式为 （注意这里 \\(B\\) 的维度虽然是 \\(n\times d\\)，但它只是 \\(b\\) 在批量方向上的重复，也就是说它的独立变量仍然是 \\(n\\) 个。）

$$
    Y = W X + B \qquad \qquad (9)
  $$

同样，张量形式如下

$$
  y_{ij} = \sum_{k=1}^m w_{jk} x_{ki} + b_{i}
  \qquad \qquad (10)
  $$

损失对权重矩阵的梯度为 

$$
  \frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial W}
  \qquad \qquad (11)
  $$

然后再用张量表示各项

$$
  \begin{aligned}
  R &= \frac{\partial L}{\partial W} = r_{ij} e_i e_j \\
  P &= \frac{\partial L}{\partial Y} = p_{ij} e_i e_j, \quad p_{ij} = \frac{\partial L}{\partial y_{ij}}\\
  Q &= \frac{\partial Y}{\partial W} = q_{ijst} e_i e_j e_s e_t, \quad q_{ijst} = \frac{\partial y_{ij}}{\partial w_{st}}
  \end{aligned}
  \qquad \qquad (12)
  $$

将以上各项带入式 (11)，则有

$$
  \begin{aligned}
    &r_{st}e_s e_t = p_{ij} e_i e_j \cdot q_{ijst} e_i e_j e_s e_t = p_{ij} q_{ijst} e_s e_t\\
    \Rightarrow & r_{st} = p_{ij} q_{ijst}
  \end{aligned}
  \qquad \qquad (13)
  $$

为了更好地计算 \\(\frac{\partial y_{ij}}{\partial w_{st}}\\)，我们把公式 (10) 稍微展开一下

$$
  y_{ij} = w_{j1}x_{1i} + w_{j2}x_{2i} + ... + w_{jm}x_{mi} + b_{ij}
  \qquad \qquad (14)
  $$

可以发现，对于 \\(w_{st}\\) 来说，当 \\(s = j, t=1\\) 时， \\(\frac{\partial y_{ij}}{\partial w_{st}} = x_{1i}\\)，当 \\(s=j, t=2\\) 时，\\(\frac{\partial y_{ij}}{\partial w_{st}} = x_{2i}\\)。依次类推，不难看到，当 \\(s=j\\) 时，\\(\frac{\partial y_{ij}}{\partial w_{st}} = x_{ti}\\)，而所有其他情况下都有 \\(\frac{\partial y_{ij}}{\partial w_{st}} = 0\\)。所以 

$$
  q_{ijst} = \frac{\partial y_{ij}}{\partial w_{st}} = \left\{\begin{aligned}
  x_{ti} \quad \mathrm{if} \quad & s = j \\
  0 \quad \mathrm{if} \quad & s \neq j
\end{aligned}\right.
\qquad \qquad (15)
  $$

再将上式代入 (13) 可得

$$
  r_{st}  = p_{is} x_{ti}
  \qquad \qquad (16)
  $$

还原成矩阵形式就为

$$
  \frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial Y}\right)^{\top} X^{\top}
  \qquad \qquad (17)
  $$

对于偏置项来说，损失对它的梯度为 

$$
  \frac{\partial L}{\partial b} = \frac{\partial L}{\partial Y} \frac{\partial Y}{\partial b}
  \qquad \qquad (18)
  $$

照样可以写成张量形式

$$
  \begin{aligned}
  &\frac{\partial L}{\partial b_k} e_k =  \frac{\partial L}{\partial y_{ij}} e_i e_j \cdot  \frac{\partial y_{ij}}{\partial b_k} e_i e_j e_k = \frac{\partial L}{\partial y_{ij}} \frac{\partial y_{ij}}{\partial b_k} e_k \\
  \Rightarrow & \frac{\partial L}{\partial b_k} = \frac{\partial L}{\partial y_{ij}} \frac{\partial y_{ij}}{\partial b_k}
  \end{aligned}
  \qquad \qquad (19)
  $$

根据公式 (10) 可得

$$
   \frac{\partial y_{ij}}{\partial b_k} = \left\{\begin{aligned}
  &1 \quad \mathrm{if} \quad k = i \\
  &0 \quad \quad otherwise
   \end{aligned}
   \right.
   \qquad \qquad (20)
  $$

再带入公式 (19)，得到

$$
  \frac{\partial L}{\partial b_k} = \sum_{i=1}^n \sum_{j=1}^b \frac{\partial L}{\partial y_{ij}} \frac{\partial y_{ij}}{\partial b_k} = \sum_{j=1}^b \frac{\partial L}{\partial y_{kj}}
  \qquad \qquad (21)
  $$

也就是说损失对偏置向量 \\(b\\) 的梯度等于矩阵 \\(\frac{\partial L}{\partial Y}\\) 在批量维度上求和得到的向量。

通过以上的推导，我们可以总结出以下结论：

1. 全连接层的前馈计算的输入为 \\(X\\)，输出为 \\(Y\\)；
2. 全连接层的反向传播需要损失 \\(L\\) 对权重矩阵 \\(W\\) 和偏置向量 \\(b\\) 的梯度，其中损失对权重矩阵的梯度计算依赖于 \\(X\\) 和 \\(\frac{\partial L}{\partial Y}\\)，损失对偏置的梯度依赖于 \\(\frac{\partial L}{\partial Y}\\)。

##### 参考文章

* https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/