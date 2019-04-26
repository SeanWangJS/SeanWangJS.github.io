---
title: 反向传播算法
---

训练集 

$$
S = \{(x^{(i)}, y^{(i)}) \mid i = 1,,,m\}
$$

前馈计算

$$
\begin{aligned}
z^{(l+1)} &= W^{(l)} a^{(l)} + b^{(l)}\\
a^{(l+1)} &= f(z^{(l+1)})
\end{aligned}
$$

其中输入层 \\(a^{(1)} = x\\)，输出层 \\(a^{(n_l)} = \hat{y}\\)

神经网络

$$
h_{W, b}(x)
$$

其中 \\(W,b\\) 分别表示网络的权重矩阵集以及偏置项集

单个样本损失函数

$$
J(W, b; x, y) = \frac 1 2 \parallel h_{W, b}(x) - y \parallel^2
$$

总体损失函数

$$
J(W, b) = \frac 1 m \sum_{i=1}^m J(W, b;x^{(i)}, y^{(i)}) + \frac {\lambda} 2 \sum_{l=1}\sum_{i=1}\sum_{j=1} \left(W_{ij}^{(l)}\right)^2
$$

梯度下降优化，迭代公式

$$
\begin{aligned}
W_{ij}^{(l)} &= W_{ij}^{(l)} - \alpha \frac{\partial }{\partial W_{ij}^{(l)}} J(W, b)\\
 b_i^{(l)} &= b_{i}^{(l)} - \alpha \frac{\partial }{\partial b_i^{(l)}} J(W, b) 
\end{aligned}
$$

<!-- 其中 -->

<!-- $$
\begin{aligned}
\frac{\partial }{\partial W_{ij}^{(l)}} J(W, b) &= \frac 1 m \sum_{k=1}^m \frac{\partial }{\partial W_{ij}^l} J(W, b; x^{(k)}, y^{(k)}) + \lambda W_{ij}^{(l)}\\
\frac{\partial }{\partial b_i^{(l)}} J(W, b)&=\frac {1} {m} \sum_{k=1}^m 
\frac{\partial}{\partial b_i^{(l)}}J(W, b; x^{(k)}, y^{(k)})
\end{aligned}
$$ -->

反向传播：
1、 前馈计算每一层的激活值 \\(a^{(l)}\\)，直到最后一层 \\(a^{(n_l)}\\)。
2、 设最后一层与倒数第二层的权重矩阵为 \\(W^{n_l-1}\\)，计算总体损失函数对各权重参数的偏导数

$$
\begin{aligned}
\frac{\partial}{\partial W_{ij}^{n_l-1}} J(W, b) 
&= \frac{\partial J}{\partial a_j^{n_l}} \frac{\partial a_{j}^{n_l}}{\partial z_j^{n_l}}\frac{\partial z_j^{n_l}}{\partial W_{ij}^{n_l-1}}\\
&=\frac 1 2\frac{\partial}{\partial a_j^{n_l}} 
 \parallel a_j^{n_l} - y^{(j)} \parallel^2 \cdot f'(z_j^{n_l}) \cdot \frac{\partial }{\partial W_{ij}^{n_l-1}} \left( \sum_{k} W_{kj}^{n_l-1} a_{k}^{n_l-1} +b^{n_l-1} \right)\\
 &=(a_j^{n_l} - y^{(j)}) f'(z_j^{n_l}) a_i^{n_l-1}
\end{aligned}
$$

3、 设倒数第二层与倒数第三层的权重矩阵为 \\(W^{n_l-2}\\)，计算总体损失函数对各权重参数的偏导数

$$
\frac{\partial}{\partial W_{ij}^{n_l-2}} J(W, b) 
= \frac{\partial J}{\partial z_{j}^{n_l-1}} \frac{\partial z_j^{n_l - 1}}{\partial W_{ij}^{n_l - 2}} = \frac{\partial J}{\partial z_{j}^{n_l-1}} a_{i}^{n_l-2}  
$$

其中 

$$
\begin{aligned}
\frac{\partial J}{\partial z_{j}^{n_l-1}}
&= \sum_{k=1}\frac{\partial J}{\partial z_{k}^{n_l}}\frac{\partial z_{k}^{n_l}}{\partial z_{j}^{n_l - 1}}\\
&=\sum_{k=1}
\frac{\partial J}{\partial a_{k}^{n_l}}
\frac{\partial a_{k}^{n_l}}{\partial z_{k}^{n_l}}
\cdot
\frac{\partial z_{k}^{n_l}}{\partial a_{j}^{n_l - 1}}
\frac{\partial a_{j}^{n_l - 1}}{\partial z_{j}^{n_l - 1}}\\
&=\sum_{k=1}(a_k^{n_l} - y^{(k)}) f'(z_k^{n_l}) W_{jk}^{n_l - 1} f'(a_j^{n_l-1})
\end{aligned}
$$

于是 

$$
\frac{\partial}{\partial W_{ij}^{n_l-2}} J(W, b) 
=\sum_{k=1}(a_k^{n_l} - y^{(k)}) f'(z_k^{n_l}) W_{jk}^{n_l - 1} f'(a_j^{n_l-1})a_{i}^{n_l-2}  
$$

4、 定义\\(\delta_{i}^{n_l}=(a_i^{n_l} - y^{(i)})f'(z_i^{n_l})\\)， 那么

$$
\frac{\partial}{\partial W_{ij}^{n_l-1}} J(W, b)  = \delta_{j}^{n_l} a_i^{n_l-1}
$$

$$
\frac{\partial}{\partial W_{ij}^{n_l-2}} J(W, b) = \sum_{k=1}\delta_{k}^{n_l} W_{jk}^{n_l - 1} f'(a_j^{n_l-1})a_{i}^{n_l-2}  
$$

若再定义 \\(\delta_j^{n_l-1} = \sum_{k=1}\delta_{k}^{n_l} W_{jk}^{n_l - 1} f'(a_j^{n_l-1})\\)， 那么则有

$$
\frac{\partial}{\partial W_{ij}^{n_l-2}} J(W, b) =\delta_j^{n_l-1} a_{i}^{n_l-2}  
$$

于是可以归纳，对于任意的层 \\(l\\)

$$
\frac{\partial J}{\partial W_{ij}^l} = \delta_{j}^{l+1}a_i^l
$$

并且

$$
\delta_j^l = \sum_{k=1}\delta_{k}^{l+1} W_{jk}^l f'(a_j^l)
$$

<!-- 2、 对于输出层的每个神经元，计算

$$
\begin{aligned}
\delta_i^{(n_l)} &= {\frac{\partial}{\partial z_{i}^{(n_l)}}} J(W, b; x^{(i)}, y^{(i)})\\
&= \frac{\partial J}{\partial f(z_i^{(n_l)})} \frac{\partial f(z_i^{(n_l)})}{\partial z_{i}^{(n_l)}}\\
&=\frac{\partial J}{\partial a_i^{n_l}} f'(z_i^{n_l})\\
&=\frac{\partial}{\partial a_i^{n_l}} 
\frac 1 2 \parallel a_i^{n_l} - y^{(i)} \parallel^2
f'(z_i^{n_l})\\
&=(a_i^{n_l} - y^{(i)})f'(z_i^{n_l})
\end{aligned}
$$
 -->















