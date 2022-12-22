---
title: 卷积层的反向传播分析
tags: 神经网络 反向传播
---

卷积层是卷积神经网络最基本的结构，在以前的文章中，我们讨论了卷积层的前馈计算方法，而神经网络的学习过程包括前馈计算和梯度的反向传播两个部分，本文就准备对卷积层的梯度计算进行分析。为了简单起见，我们使用一个 `3x3` 的输入张量和 `2x2` 的卷积核来举例说明，并把结论推广的任意大小输入和卷积核情况。

$$
\left[\begin{aligned}
x_{11} \quad x_{12}\quad x_{13}\\
x_{21} \quad x_{22}\quad x_{23}\\
x_{31} \quad x_{32}\quad x_{33}
\end{aligned}
\right] \star \left[\begin{aligned}
k_{11} \quad k_{12}\\
k_{21} \quad k_{22}
\end{aligned}
\right] = \left[\begin{aligned}
y_{11} \quad y_{12}\\
y_{21} \quad y_{22}
\end{aligned}
\right]
$$

如果使用 `im2col` 把输入张量转换成列形式矩阵，则上述卷积又可以表示成矩阵乘法的形式

$$
\left[\begin{aligned}
x_{11} \quad x_{12} \quad x_{21} \quad x_{22}\\
x_{12} \quad x_{13} \quad x_{22} \quad x_{23}\\
x_{21} \quad x_{22} \quad x_{31} \quad x_{32}\\
x_{22} \quad x_{23} \quad x_{32} \quad x_{33}\\
\end{aligned}
\right] \cdot \left[\begin{aligned}
k_{11} \\ k_{12} \\ k_{21} \\ k_{22}\\
\end{aligned}
\right] = \left[\begin{aligned}
y_{11} \\ y_{12} \\ y_{21} \\ y_{22}\\
\end{aligned} \right]
$$

稍微求解一下，可以得到展开的形式

$$
  \begin{aligned}
  y_{11} &= k_{11} x_{11} + k_{12} x_{12} + k_{21} x_{21} + k_{22} x_{22}\\
  y_{12} &= k_{11} x_{12} + k_{12} x_{13} + k_{21} x_{22} + k_{22} x_{23}\\
  y_{21} &= k_{11} x_{21} + k_{12} x_{22} + k_{21} x_{31} + k_{22} x_{32}\\
  y_{22} &= k_{11} x_{22} + k_{12} x_{23} + k_{21} x_{32} + k_{22} x_{33}\\
  \end{aligned} \qquad \qquad (1)
  $$

设损失为 \\(L\\)，则损失对卷积核的梯度根据复合函数的求导规则可以写为

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

然后我们再将其写为矩阵形式

$$
\left[\frac{\partial L}{\partial y_{11}} \quad \frac{\partial L}{\partial y_{12}}\quad \frac{\partial L}{\partial y_{21}} \quad \frac{\partial L}{\partial y_{22}}\right] \cdot
\left[
\begin{matrix}
  x_{11} & x_{12} & x_{21} & x_{22}\\
  x_{12} & x_{13} & x_{22} & x_{23}\\
  x_{21} & x_{22} & x_{31} & x_{32}\\
  x_{22} & x_{23} & x_{32} & x_{33}
\end{matrix}
\right] = \left[
  \begin{aligned}
  \frac{\partial L}{\partial k_{11}} \\
  \frac{\partial L}{\partial k_{12}} \\
  \frac{\partial L}{\partial k_{21}} \\
  \frac{\partial L}{\partial k_{22}}
  \end{aligned}
\right]
$$

从上式不难发现，一旦我们知晓了损失函数对输出的梯度，则可以通过矩阵乘法计算损失函数对卷积核权重的梯度，即

$$
\frac{\partial L}{\partial K} = \frac{\partial L}{\partial Y} \cdot \mathbf{im2col}(X)
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
  \end{aligned} \qquad\qquad (2)
  $$

仅从上式来说，还看不出来明显的规律，若考虑以下乘积

$$
\left[
  \begin{aligned}
  \frac{\partial L}{\partial y_{11}} \\
  \frac{\partial L}{\partial y_{12}} \\
  \frac{\partial L}{\partial y_{21}} \\
  \frac{\partial L}{\partial y_{22}}
  \end{aligned}
\right] \cdot
\left[
  \begin{aligned}
  k_{11} \quad k_{12} \quad k_{21} \quad k_{22}
  \end{aligned}
\right] = 
\left[
  \begin{aligned}
  \frac{\partial L}{\partial y_{11}}k_{11} \quad \frac{\partial L}{\partial y_{11}}k_{12}\quad \frac{\partial L}{\partial y_{11}}k_{21} \quad \frac{\partial L}{\partial y_{11}}k_{22}\\
  \frac{\partial L}{\partial y_{12}}k_{11} \quad \frac{\partial L}{\partial y_{12}}k_{12}\quad \frac{\partial L}{\partial y_{12}}k_{21} \quad \frac{\partial L}{\partial y_{12}}k_{22}\\
  \frac{\partial L}{\partial y_{21}}k_{11} \quad \frac{\partial L}{\partial y_{21}}k_{12}\quad \frac{\partial L}{\partial y_{21}}k_{21} \quad \frac{\partial L}{\partial y_{21}}k_{22}\\
  \frac{\partial L}{\partial y_{22}}k_{11} \quad \frac{\partial L}{\partial y_{22}}k_{12}\quad \frac{\partial L}{\partial y_{22}}k_{21} \quad \frac{\partial L}{\partial y_{22}}k_{22}
  \end{aligned}
\right] \qquad \qquad (3)
$$

以及 `im2col(X)` 的结果

$$
\left[
\begin{matrix}
  x_{11} & x_{12} & x_{21} & x_{22}\\
  x_{12} & x_{13} & x_{22} & x_{23}\\
  x_{21} & x_{22} & x_{31} & x_{32}\\
  x_{22} & x_{23} & x_{32} & x_{33}
\end{matrix}
\right]
$$

再观察公式(2)，可以发现，将公式(3)的结果按 `im2col` 的逆过程折叠，折叠到相同位置的元素相加，即可得到 \\(\frac{\partial K}{\partial X}\\)，也就是说

\[
  \frac{\partial L}{\partial X} = \mathbf{col2im}\left(\frac{\partial L}{\partial Y} \cdot K\right)
  \]

最后总结一下，为了实现卷积运算的反向传播，我们只需要给出损失对输出张量的梯度，即可使用矩阵乘法配合 `im2col` 和 `col2im` 计算损失对输入张量和卷积核的梯度。

##### 参考文章

* [Convolutions and Backpropagations](https://pavisj.medium.com/convolutions-and-backpropagations-46026a8f5d2c)