---
title: 卷积层的反向传播分析
tags: 神经网络 反向传播
---

<!-- 在机器学习中，梯度下降是一种十分重要的优化算法。考虑模型 \\(f(x;\theta)\\)，对于任意输入 \\((x_i, y_i)\\)，模型预测值 \\(y_i' = f(x_i;\theta)\\) 与真实值 \\(y_i\\) 之间存在损失 \\(l(y_i, y_i')\\)，为了优化模型参数，可以考虑如下迭代过程

$$
\theta^{\{k+1\}} = \theta^{\{k\}} - \eta \frac{\partial l}{\partial \theta^{\{k\}}}
$$

这就是基础的梯度下降定义。现在，让我们考虑神经网络模型，与普通模型不同的是，神经网络的结构是一层一层的，每层都有独立的参数集，所以可以把它的公式大致写成下面这样

$$
net(x; \Theta) = f_n(f_{n-1}(...f_2(f_1(x;\theta_1);\theta_2)...;\theta_{n-1}); \theta_n)
$$

理论上来说，我们可以针对总的参数集合 \\(\Theta\\) 来做优化，但其中的难度相当大，一种更好的方法是对每一层的参数单独进行优化，也就是让损失 \\(l\\) 对 \\(\theta_n, \theta_{n-1},,\\) 直到 \\(\theta_1\\) 逐个求梯度并迭代更新。对于 \\(\theta_n\\) 来说，由于 \\(f_n\\) 是关于它的函数，且损失 \\(l\\) 是关于 \\(f_n\\) 的函数，所以损失对 \\(\theta_n\\) 的梯度用复合函数求导的方法可以表示为

$$
\frac{\partial l}{\partial \theta_n} = \frac{\partial l}{\partial f_n} \frac{\partial f_n}{\partial \theta_n}
$$

而对于更前面的层，求损失对参数的梯度是类似的

$$
\frac{\partial l}{\partial \theta_{n-1}} = \frac{\partial l}{\partial f_n} \frac{\partial f_n}{\partial f_{n-1}} \frac{\partial f_{n-1}}{\partial \theta_{n-1}}
$$

$$
\frac{\partial l}{\partial \theta_1} =  \frac{\partial l}{\partial f_n} \frac{\partial f_n}{\partial f_{n-1}} ... \frac{\partial f_1}{\partial \theta_1}
$$ -->

卷积层是卷积神经网络最基本的结构，在以前的文章中，我们讨论了卷积层的前馈计算方法，而神经网络的学习过程包括前馈计算和梯度的反向传播两个部分，本文就准备对卷积层的梯度计算进行分析。为了简单起见，我们使用一个 `3x3` 的输入张量和 `2x2` 的卷积核来举例说明，并把结论推广的任意大小输入和卷积核情况。

![](/resources/2022-07-20-conv-backpropagation/conv_conv.png)

图中所示卷积计算过程如下

$$
  \begin{aligned}
  y_{11} &= k_{11} x_{11} + k_{12} x_{12} + k_{21} x_{21} + k_{22} x_{22}\\
  y_{12} &= k_{11} x_{12} + k_{12} x_{13} + k_{21} x_{22} + k_{22} x_{23}\\
  y_{21} &= k_{11} x_{21} + k_{12} x_{22} + k_{21} x_{31} + k_{22} x_{32}\\
  y_{22} &= k_{11} x_{22} + k_{12} x_{23} + k_{21} x_{32} + k_{22} x_{33}\\
  \end{aligned} \qquad \qquad (1)
  $$

设损失为 \\(L\\)，则损失对卷积核的梯度为

$$
  \begin{aligned}
  \frac{\partial L}{\partial k_{11}} 
  &= \frac{\partial L}{\partial y_{11}} \frac{\partial y_{11}}{\partial k_{11}} +
  \frac{\partial L}{\partial y_{12}} \frac{\partial y_{12}}{\partial k_{11}} +
  \frac{\partial L}{\partial y_{21}} \frac{\partial y_{21}}{\partial k_{11}}+
  \frac{\partial L}{\partial y_{22}} \frac{\partial y_{22}}{\partial k_{11}} \\
  \frac{\partial L}{\partial k_{12}} 
  &= \frac{\partial L}{\partial y_{11}} \frac{\partial y_{11}}{\partial k_{12}} +
  \frac{\partial L}{\partial y_{12}} \frac{\partial y_{12}}{\partial k_{12}} +
  \frac{\partial L}{\partial y_{21}} \frac{\partial y_{21}}{\partial k_{12}}+
  \frac{\partial L}{\partial y_{22}} \frac{\partial y_{22}}{\partial k_{12}} \\
  \frac{\partial L}{\partial k_{21}} 
  &= \frac{\partial L}{\partial y_{11}} \frac{\partial y_{11}}{\partial k_{21}} +
  \frac{\partial L}{\partial y_{12}} \frac{\partial y_{12}}{\partial k_{21}} +
  \frac{\partial L}{\partial y_{21}} \frac{\partial y_{21}}{\partial k_{21}}+
  \frac{\partial L}{\partial y_{22}} \frac{\partial y_{22}}{\partial k_{21}} \\
  \frac{\partial L}{\partial k_{22}} 
  &= \frac{\partial L}{\partial y_{11}} \frac{\partial y_{11}}{\partial k_{22}} +
  \frac{\partial L}{\partial y_{12}} \frac{\partial y_{12}}{\partial k_{22}} +
  \frac{\partial L}{\partial y_{21}} \frac{\partial y_{21}}{\partial k_{22}}+
  \frac{\partial L}{\partial y_{22}} \frac{\partial y_{22}}{\partial k_{22}}
  \end{aligned}
  $$

利用 \\(y\\) 对 \\(k\\) 的导数简化一下，可得

$$
   \begin{aligned}
  \frac{\partial L}{\partial k_{11}} 
  &= \frac{\partial L}{\partial y_{11}}x_{11} +
  \frac{\partial L}{\partial y_{12}} x_{12} +
  \frac{\partial L}{\partial y_{21}} x_{21}+
  \frac{\partial L}{\partial y_{22}} x_{22} \\
  \frac{\partial L}{\partial k_{12}} 
  &= \frac{\partial L}{\partial y_{11}} x_{12} +
  \frac{\partial L}{\partial y_{12}} x_{13} +
  \frac{\partial L}{\partial y_{21}} x_{22}+
  \frac{\partial L}{\partial y_{22}} x_{23} \\
  \frac{\partial L}{\partial k_{21}} 
  &= \frac{\partial L}{\partial y_{11}} x_{21} +
  \frac{\partial L}{\partial y_{12}} x_{22} +
  \frac{\partial L}{\partial y_{21}} x_{31}+
  \frac{\partial L}{\partial y_{22}} x_{32} \\
  \frac{\partial L}{\partial k_{22}} 
  &= \frac{\partial L}{\partial y_{11}} x_{22} +
  \frac{\partial L}{\partial y_{12}} x_{23} +
  \frac{\partial L}{\partial y_{21}} x_{32}+
  \frac{\partial L}{\partial y_{22}} x_{33}
  \end{aligned}
  $$

将其于公式 (1) 对比一下不难发现，上式也是一个卷积运算，只不过它的卷积核变成了损失对 \\(y\\) 的梯度，如下图所示

![](/resources/2022-07-20-conv-backpropagation/conv_gradient-kernel.png)

令输入张量为 \\(X\\)，输出张量为 \\(Y\\)，卷积核为 \\(K\\)，则损失对卷积核的梯度就可以表示为

$$
  \frac{\partial L}{\partial K} = \mathrm{conv}(X, \frac{\partial L}{\partial Y})
  $$

接下来我们考虑损失对输入张量的梯度计算，根据复合函数求导规则，我们有下式

$$
  \frac{\partial L}{\partial x_i} = \sum_{j} \frac{\partial L}{\partial y_j} \frac{\partial y_j}{\partial x_i}
$$

其中 \\(x_i\\) 表示输入张量的某个元素，把上式展开的话具体如下

$$
  \begin{aligned}
  \frac{\partial L}{\partial x_{11}} &= \frac{\partial L}{\partial y_{11} } k_{11}\\
  \frac{\partial L}{\partial x_{12}} &= \frac{\partial L}{\partial y_{11} } k_{12} +  \frac{\partial L}{\partial y_{12}} k_{11} \\ 
  \frac{\partial L}{\partial x_{13}} &= \frac{\partial L}{\partial y_{12}}k_{12}\\
  \frac{\partial L}{\partial x_{21}} &= \frac{\partial L}{\partial y_{11} }k_{21} +
  \frac{\partial L}{\partial y_{21}}k_{11} \\
  \frac{\partial L}{\partial x_{22}} &= \frac{\partial L}{\partial y_{11} }k_{22} +
  \frac{\partial L}{\partial y_{12}}k_{21} +
  \frac{\partial L}{\partial y_{21}}k_{12} +
  \frac{\partial L}{\partial y_{22}}k_{11}\\
  \frac{\partial L}{\partial x_{23}} &= \frac{\partial L}{\partial y_{12}} k_{12} +
  \frac{\partial L}{\partial 22} k_{22} \\ 
  \frac{\partial L}{\partial x_{31}} &= \frac{\partial L}{\partial y_{21}} k_{21}\\ 
  \frac{\partial L}{\partial x_{32}} &= \frac{\partial L}{\partial y_{21}} k_{22} + \frac{\partial L}{\partial y_{22}} k_{21} \\
  \frac{\partial L}{\partial x_{33}} &= \frac{\partial L}{\partial y_{22}} k_{22}
  \end{aligned}
  $$

虽然不太明显，但上式其实是 \\(\frac{\partial L}{\partial Y}\\) 与 \\(K\\) 的反卷积运算，也就是下图所示的运算关系

![](/resources/2022-07-20-conv-backpropagation/conv_gradient-x.png)

需要注意的是，这里的卷积核其实是 \\(K\\) 旋转了 180° 之后的结果。用公式表示的话如下

$$
  \frac{\partial L}{\partial X} = \mathrm{deconv}\left(\frac{\partial L}{\partial Y}, rotate(K)\right) 
  $$

最后总结一下，为了实现卷积运算的反向传播，我们只需要给出损失对输出张量的梯度，即可使用卷积和反卷积方法计算损失对输入张量和卷积核的梯度。