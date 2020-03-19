---
title: 图像噪点水平估计
tags: 图像处理
---

给定一张图片，大小为 \(M\times N \times c\)，将其分割成 \(d\times d \times c\) 的小块，其中 c 是颜色通道数，可以得到总数为 \(s = (M - d + 1) \times (N - d + 1)\) 个局部图片。每个局部图像的像素可以重新组织成大小为 \(r=cd^2\) 的向量 \(x_i\)，而所有局部图像的表示向量则构成矩阵

\[
  X_s = \left[
    \begin{aligned}
    x_1\\x_2\\... \\x_s
    \end{aligned}
    \right]
  \]

由于噪点是像素点数值的随机起伏，所以每个像素可以被分解成真实值 \(\hat{x}_i\)和噪声 \(e_t\)之和

\[
  x_i = \hat{x}_i + e_i
  \]

写成矩阵的形式为

\[
  X_s = \hat{X}_s + Er
  \]

如果我们从主成分分析的观点来看，\(\hat{X}_s\) 可以被看作是用 \(X_s\) 前几个主成分重建后的数据集

值得注意的是，向量 \(\hat{x}_t\) 可以被嵌入维度比 \(x_t\) 更低的子空间，即 \(m=\dim(\hat{x}_t) < r\)。假如我们把无噪声的理想图片看作绝对平静的水面，那么噪声就相当于风扬起的涟漪，显然平面是一个二维空间，但是要表达波动的水面则需要三维向量。

所有的向量 \(x_t\) 构成了我们的数据集 \(X_s = \{x_t\}_{t=1}^s\)，为了建立噪声的模型，这里使用零均值的多元高斯分布，即 

\[
  e_t \thicksim N_r(0, \sigma^2 \mathbf{I})
  \]

也就是说 \(e_t\) 中的每个元素独立同分布，并且方差为 \(\sigma^2\)。从这个模型来看，方差越大，噪点越多，于是方差就可以作为噪点水平的一个评价指标，而这篇文章的目的也就是找到这个方差的值。

将 \(x_t\) 的分解重新写成

\[
  x_t = A y_t + e_t
  \]

即 \(\hat{x}_t = Ay_t\)，其中 \(A \in \mathbb{R}^{r\times m}\)，并且 \(A^T A = \mathbf{I}\)，它的所有行向量张成空间 \(\mathbb{R}^m\)，反过来又有 \(y_t = A^T \hat{x}_t\)

于是 \(y_t\) 就是 \(\hat{x}_t\) 在 \(A\) 张成空间上的投影，同时也是 \(x_t\) 在\(A\) 张成空间上的投影。如果对 \(A\) 和 \(y_t\) 进行增广，得到 

\[
  x_t = R \left[\begin{aligned} 
      y  \\
      \mathbf{0}
      \end{aligned}
      \right] + e_t
  \]

其中 \(R = [A, U]\)，这里的 \(U \in \mathbb{R}^{(r - m) \times m}\) ，并且满足 \(R^T R = \mathbf{I}\)。将上式两端同时乘以 \(R^T\)，得到

\[
  \begin{aligned}
  R^T x_t &=R^T R \left[\begin{aligned} 
      y_t  \\
      \mathbf{0}
      \end{aligned}
      \right] + R^T e_t \\
      &= \left[
        \begin{aligned}
        y_t + A^T e_t\\
        U^T e_t
        \end{aligned}
        \right]
      \end{aligned}
  \]

