---
layout: post
title: 条件高斯分布
tags: 概率论 高斯分布 正态分布 模式识别系列
---

最初，均值为 $$\mu$$ ，协方差矩阵为 \\(\Sigma\\) 的多元高斯分布具有如下形式

$$
p(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} \exp\left(
-\frac 1 2 (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)
\right)
$$

为了讨论简单起见，我们先考虑变量为 \\(x, y\\) 的二元高斯分布，此时均值和协方差矩阵分别为

$$
\mu^T = (\mu_1\quad \mu_2)^T\\
\Sigma =
\left[
\begin{aligned}
\sigma_{11} &\quad \sigma_{12}\\\sigma_{21}&\quad\sigma_{22}
\end{aligned}
\right]
$$

其中 \\(\sigma_{12} = \sigma_{21}\\)。为了表达方便，这里假设了设协方差矩阵的逆矩阵

$$
\Sigma^{-1} =
\left[
\begin{aligned}
\lambda_{11} &\quad \lambda_{12}\\\lambda_{21}&\quad\lambda_{22}
\end{aligned}
\right]
$$

其中 \\(\lambda_{12} = \lambda_{21}\\)。于是高斯分布的表达式就展开成了下述形式

$$
p(x, y) =  \frac{1}{2\pi \sqrt{|\Sigma|}} \exp\left(-\frac 1 2
\lambda_{11}(x-\mu_1)^2 -\frac 1 2 \lambda_{22} (y-\mu_2)^2 + \lambda_{12}(x -\mu_1)(y-\mu_2
)
\right)
$$

### 边缘分布——二维情况

为了计算多元高斯分布的边缘分布，我们需要对其余变量在全空间上积分。也就是说

$$
p(x) = \int_{-\infty}^{\infty} p(x, y)\mathrm{d}y
$$

为了解析地求得这个积分，我们先把联合分布 \\(p(x, y)\\) 的指数项展开

$$
\begin{aligned}
&-\frac 1 2
\left(\lambda_{11}(x-\mu_1)^2 + \lambda_{22} (y-\mu_2)^2 -2 \lambda_{12}(x -\mu_1)(y-\mu_2)\right)\\
&= -\frac 1 2
\left(
\lambda_{11} x^2 -2\lambda_{11}x\mu_1 + \lambda_{11}\mu_1^2+ \lambda_{22}y^2 -2\lambda_{22} y\mu_2 +\lambda_{22}\mu_2^2 
-2\lambda_{12}xy + 2\lambda_{12} x\mu_2 + 2\lambda_{12} y\mu_1 - 2\lambda_{12}\mu_1 \mu_2
\right)\\
&=(-\frac 1 2 \lambda_{11}x^2 + \lambda_{11}\mu_1x -\lambda_{12}\mu_2 x)
+(-\frac 1 2 \lambda_{22}y^2 + \lambda_{22}\mu_2 y + \lambda_{12} x y - \lambda_{12} \mu_1  y)\\
&-\frac 1 2(\lambda_{11}\mu_1^2+\lambda_{22}\mu_2^2  -2 \lambda_{12}\mu_1 \mu_2)\\
&= C(x) + C(y) + C
\end{aligned}
$$

为了简化表达，我们令 \\(C(x), C(y), C\\) 来分别表达含有 *x*，*y* 的项和常数项。
由于是对 *y* 的积分，所以与 *y* 无关的项可以提到积分符号外面去，即得到

$$
p(x) = \frac{1}{2\pi \sqrt{\mid\Sigma\mid}} \exp(C(x)+C) \int_{-\infty}^{\infty} e^{C(y)} \mathrm{d}y
$$

为了计算[高斯积分](https://en.wikipedia.org/wiki/Gaussian_integral)，我们先对 \\(C(y)\\) 进行配平方

$$
\begin{aligned}
C(y) &= -\frac 1 2 \lambda_{22}y^2 + \lambda_{22}\mu_2 y + \lambda_{12} x y - \lambda_{12} \mu_1  y\\
&=-\frac 1 2 \lambda_{22}\left(
y^2 - \frac{2(\lambda_{22}\mu_2 +\lambda_{12}x -\lambda_{12}\mu_1)}{\lambda_{22}}y
\right)\\
&=-\frac 1 2 \lambda_{22} \left(
y-\frac{\lambda_{22}\mu_2 +\lambda_{12}x -\lambda_{12}\mu_1}{\lambda_{22}}
\right)^2 + \frac{(\lambda_{22}\mu_2 +\lambda_{12}x -\lambda_{12}\mu_1)^2}{2\lambda_{22}}\\
&=-z^2 + C_2(x)
\end{aligned}
$$

同样，为了简化表达，上式中令 \\(z = \sqrt{\frac{\lambda_{22}}{2}} \left(y-\frac{\lambda_{22}\mu_2 +\lambda_{12}x -\lambda_{12}\mu_1}{\lambda_{22}}\right)\\)，第二项又与 *y* 无关。
于是我们看到

$$
\begin{aligned}
p(x) &= \frac 1 {2\pi \sqrt{|\Sigma|}}\exp(C(x) + C + C_2(x))\frac {\sqrt{2}} {\sqrt{\lambda_{22}}}\int_{-\infty}^{\infty} e^{-z^2}\mathrm{d}z\\
&= \frac 1 {\sqrt{2\lambda_{22}\pi|\Sigma|}}\exp(C(x) + C + C_2(x))
\end{aligned}
$$

仍然单独考虑指数项

$$
\begin{aligned}
C(x) +C_2(x) +C &= -\frac 1 2 \lambda_{11}x^2 + \lambda_{11}\mu_1x -\lambda_{12}\mu_2 x
+\frac{(\lambda_{22}\mu_2 +\lambda_{12}x -\lambda_{12}\mu_1)^2}{2\lambda_{22}}
+C\\
&= -\frac 1 2 (\lambda_{11} - \frac{\lambda_{12}^2}{\lambda_{22}})x^2 + (\lambda_{11}\mu_1 -\lambda_{12}\mu_2 +\frac{\lambda_{12}}{\lambda_{22}}(\lambda_{22}\mu_2 - \lambda_{12}\mu_1)) x\\
&+\frac{(\lambda_{22}\mu_2  -\lambda_{12}\mu_1)^2}{2\lambda_{22}}
+C\\
&= -\frac{\lambda_{11} \lambda_{22} - \lambda_{12}^2}{2\lambda_{22}}x^2+ \frac{\lambda_{11}\lambda_{22} - \lambda_{12}^2}{\lambda_{22}}\mu_1 x + C' +C\\
&=- \frac{\lambda_{11} \lambda_{22} - \lambda_{12}^2}{2\lambda_{22}}(x^2 -2 \mu_1 x) +C' +C\\
&= - \frac{\lambda_{11} \lambda_{22} - \lambda_{12}^2}{2\lambda_{22}}(x-\mu_1)^2 + \frac{\lambda_{11} \lambda_{22} - \lambda_{12}^2}{2\lambda_{22}}\mu_1^2 +C'+C\\
&= - \frac{\lambda_{11} \lambda_{22} - \lambda_{12}^2}{2\lambda_{22}}(x-\mu_1)^2 + C'' +C'+C
\end{aligned}
$$

根据逆矩阵行列式的性质

$$
|\Sigma|^{-1} = |\Sigma^{-1}| = \lambda_{11}\lambda_{22} - \lambda_{12}^2
$$

所以

$$
C(x) +C_2(x)+C = -\frac{(x - \mu_1)^2}{2\lambda_{22}|\Sigma|} +D
$$

其中 \\(D = C''+C'+C\\)，如果仔细算一算，会发现其实 \\(D = 0\\)。所以我们看到

$$
p(x) = \frac{1}{\sqrt{2\pi} (\lambda_{22}|\Sigma|)^{\frac 1 2}} \exp(-\frac{(x - \mu_1)^2}{2\lambda_{22}|\Sigma|} )
$$

也就是说，*x* 的分布为以 \\(\mu_1\\) 为均值，\\(\lambda_{22}\mid\Sigma\mid\\) 为方差的高斯分布。

我们仔细考察一下边缘分布的均值与方差，容易发现，其均值实际上刚好等于联合分布下 *x* 分量的均值。然后我们再简单计算一下协方差矩阵的逆矩阵

$$
\Sigma^{-1} =\frac{1}{|\Sigma|} \left[
\begin{aligned}
\sigma_{22} &\quad -\sigma_{12}\\-\sigma_{21}&\quad\sigma_{11}
\end{aligned}
\right]
$$

于是可以得到 \\(\lambda_{22} = \frac{\sigma_{11}}{\mid\Sigma\mid}\\)，那么边缘分布的方差其实就等于 \\(\sigma_{11}\\)，而这其实就是原联合分布下 *x* 分量的方差。最终可得

$$
p(x) = \frac{1}{\sqrt{2\pi  \sigma_{11}} } \exp\left(-\frac{(x - \mu_1)^2}{2\sigma_{11}} \right)
$$

从上面的分析，我们看到一个事实：对于二元高斯分布，其边缘分布是一个高斯分布，并且它的均值和方差就是原联合分布对应分量的均值和方差。

这是一个很强的结论，它让我们可以不必通过计算，直接通过联合分布的表达式写出边缘分布的表达式，下面我们试着将其推广到更高阶的情形。

### 边缘分布——高维情况

设 d 维随机向量

$$
\mathbf{z} = \left(
\begin{aligned}
\mathbf{x} \\ \mathbf{y}
\end{aligned}
  \right)
$$

其中 \\(\mathbf{x},\mathbf{y}\\) 都是向量，它们的维数和等于 d，并且 \\(\mathbf{z}\\) 的分布为高斯分布，即

$$
p(\mathbf{x},\mathbf{y})=p(\mathbf{z}) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} \exp\left(
-\frac 1 2 (\mathbf{z} - \mu)^T \Sigma^{-1} (\mathbf{z} - \mu)
\right)
$$

根据 \\(\mathbf{x},\mathbf{y}\\) 对 \\(\mathbf{z}\\) 的分割，将均值和协方差矩阵按如下分割

$$
\mu =  \left(
\begin{aligned}
\mu_{x} \\ \mu_{y}
\end{aligned}
  \right)\\

  \Sigma = \left[
  \begin{aligned}
  \Sigma_{xx}\quad & \Sigma_{xy} \\
  \Sigma_{yx}\quad & \Sigma_{yy}
  \end{aligned}
    \right]
$$

由于协方差矩阵的对称性，我们知道 \\(\Sigma_{xy} = \Sigma_{yx}^T\\)。为了表达的方便，我们再假设协方差矩阵的逆矩阵为

$$
\Sigma^{-1} = \left[
\begin{aligned}
\Lambda_{xx}\quad & \Lambda_{xy} \\
\Lambda_{yx}\quad & \Lambda_{yy}
\end{aligned}
  \right]
$$

其中 \\(\Lambda_{xy}^T = \Lambda_{yx}\\)。利用上述分解，我们可以将联合分布的指数项写成

$$
\begin{aligned}
&-\frac 1 2 ((\mathbf{x} - \mu_x)^T\quad (\mathbf{y} - \mu_y)^T)
\left[
\begin{aligned}
\Lambda_{xx}\quad & \Lambda_{xy} \\
\Lambda_{yx}\quad & \Lambda_{yy}
\end{aligned}
  \right]
  \left(
  \begin{aligned}
  \mathbf{x} - \mu_x\\
  \mathbf{y} - \mu_y
  \end{aligned}
    \right)\\
    &=-\frac 1 2
(\mathbf{x} - \mu_x)^T \Lambda_{xx}(\mathbf{x} - \mu_x)
-\frac 1 2(\mathbf{y} - \mu_y)^T \Lambda_{yy}(\mathbf{y} - \mu_y)\\
&+\frac 1 2(\mathbf{x} - \mu_x)^T \Lambda_{xy}(\mathbf{y} - \mu_y)
+\frac 1 2(\mathbf{y} - \mu_y)^T \Lambda_{yx}(\mathbf{x} - \mu_x)\\
&=-\frac 1 2 \mathbf{x}^T \Lambda_{xx} \mathbf{x} + \mu_{x}^T \Lambda_{xx} \mathbf{x} -\frac 1 2 \mu_x^T \Lambda_{xx} \mu_x\\
&-\frac 1 2 \mathbf{y}^T \Lambda_{yy} \mathbf{y} + \mu_{y}^T \Lambda_{yy} \mathbf{y} -\frac 1 2 \mu_y^T \Lambda_{yy} \mu_y\\
&+\mathbf{x}^T \Lambda_{xy} \mathbf{y}-\mu_x^T \Lambda_{xy} \mathbf{y}-\mu_y^T \Lambda_{yx} \mathbf{x} + \mu_x^T \Lambda_{xy} \mu_y \\
&= \left(
-\frac 1 2 \mathbf{x}^T \Lambda_{xx} \mathbf{x} + (\mu_x^T \Lambda_{xx} - \mu_y^T \Lambda_{yx})\mathbf{x}
  \right)
  +\left(
-\frac 1 2 \mathbf{y}^T \Lambda_{yy} \mathbf{y} +(\mu_{y}^T \Lambda_{yy} - \mu_x^T \Lambda_{xy} + \mathbf{x}^T \Lambda_{xy})\mathbf{y}
    \right)\\
    &+\left(
 -\frac 1 2 \mu_x^T \Lambda_{xx} \mu_x-\frac 1 2 \mu_y^T \Lambda_{yy} \mu_y + \mu_x^T \Lambda_{xy} \mu_y
      \right)\\
      &=C(\mathbf{x}) +C(\mathbf{y})+C
\end{aligned}
$$

这里为了方便表达，我们仍然使用 \\(C(\mathbf{x}), C(\mathbf{y}),C\\) 来分别表示与 \\(\mathbf{x}, \mathbf{y}\\) 相关的项和无关项。

现在我们要求边缘分布 \\(p(\mathbf{x})\\)，这需要在全空间上对 \\(\mathbf{y}\\) 积分

$$
\begin{aligned}
p(\mathbf{x}) &= \int p(\mathbf{x}, \mathbf{y}) \mathrm{d}\mathbf{y}\\
&= \frac 1 {\sqrt{(2\pi)^d |\Sigma|}}e^{C(\mathbf{x})+C}\int e^{C(\mathbf{y})}\mathrm{d}\mathbf{y}
\end{aligned}
$$

对 \\(C(\mathbf{y})\\) 配平方

$$
\begin{aligned}
C(\mathbf{y}) &= -\frac 1 2 \mathbf{y}^T \Lambda_{yy} \mathbf{y} +(\mu_{y}^T \Lambda_{yy} - \mu_x^T \Lambda_{xy} + \mathbf{x}^T \Lambda_{xy})\mathbf{y}\\
&=?
\end{aligned}
$$

存在一点困难，于是我们反过来先假设配出了平方项，利用待定系数法，假设

$$
\begin{aligned}
C(\mathbf{y}) &= -\frac 1 2 (\mathbf{y} - \lambda)^T \Lambda_{yy}(\mathbf{y}- \lambda) + \eta\\
&=-\frac 1 2 \mathbf{y}^T \Lambda_{yy} \mathbf{y} + \lambda^T \Lambda_{yy} \mathbf{y} -\frac 1 2 \lambda^{T}\Lambda_{yy} \lambda + \eta
\end{aligned}
$$

于是我们看到

$$
\lambda^T \Lambda_{yy}  = \mu_{y}^T \Lambda_{yy} - \mu_x^T \Lambda_{xy} + \mathbf{x}^T \Lambda_{xy}\\
\eta = \frac 1 2 \lambda^T  \Lambda_{yy} \lambda
$$

若我们再令 \\(\mathbf{m}^T =  \mu_{y}^T \Lambda_{yy} - \mu_x^T \Lambda_{xy} + \mathbf{x}^T \Lambda_{xy}\\)，那么 \\(\lambda^T = \mathbf{m}^T\Lambda_{yy}^{-1}\\)，于是有

$$
C(\mathbf{y}) =  -\frac 1 2 (\mathbf{y} -\Lambda_{yy}^{-1}\mathbf{m})^T \Lambda_{yy}(\mathbf{y}- \Lambda_{yy}^{-1}\mathbf{m}) + \frac 1 2\mathbf{m}^T \Lambda_{yy}^{-1}\mathbf{m}
$$

如果令 \\(\mathbf{t} = \mathbf{y} -\Lambda_{yy}^{-1}\mathbf{m}\\) ， \\(C_2(\mathbf{x}) =  \frac 1 2\mathbf{m}^T \Lambda_{yy}^{-1}\mathbf{m}\\) ，那么

$$
C(\mathbf{y}) = -\frac 1 2 \mathbf{t}^T\Lambda_{yy}\mathbf{t}+C_2(\mathbf{x})
$$

将上式代入积分公式，可得

$$
p(\mathbf{x}) = \frac 1 {\sqrt{(2\pi)^d |\Sigma|}}e^{C(\mathbf{x})+C+C_2(\mathbf{x})}
\int
\exp\left(
 -\frac 1 2 \mathbf{t}^T\Lambda_{yy}\mathbf{t}
  \right)
 \mathrm{d}\mathbf{t}
$$

计算[高斯积分](https://en.wikipedia.org/wiki/Gaussian_integral)

$$
\int
\exp\left(
 -\frac 1 2 \mathbf{t}^T\Lambda_{yy}\mathbf{t}
  \right)
 \mathrm{d}\mathbf{t} = \sqrt{(2\pi)^k |\Lambda_{yy}|^{-1}}
$$

上式中的 \\(k < d\\)，是向量 *y* 的维度。于是

$$
p(\mathbf{x}) = \frac 1 {\sqrt{(2\pi)^{d-k} |\Sigma| |\Lambda_{yy}|}}e^{C(\mathbf{x})+C_2(\mathbf{x})+C}
$$

先考虑指数项

$$
\begin{aligned}
&C(\mathbf{x}) +C_2(\mathbf{x})+C\\
&= -\frac 1 2 \mathbf{x}^T \Lambda_{xx} \mathbf{x} + (\mu_x^T \Lambda_{xx} - \mu_y^T \Lambda_{yx})\mathbf{x}
 +\frac 1 2\mathbf{m}^T \Lambda_{yy}^{-1}\mathbf{m}\\
 &-\frac 1 2 \mu_x^T \Lambda_{xx} \mu_x-\frac 1 2 \mu_y^T \Lambda_{yy} \mu_y + \mu_x^T \Lambda_{xy} \mu_y\\
 &=-\frac 1 2 \mathbf{x}^T (\Lambda_{xx} - \Lambda_{xy}\Lambda_{yy}^{-1}\Lambda_{yx})\mathbf{x}
 +\mu_x^T\left(
 \Lambda_{xx} - \Lambda_{xy} \Lambda_{yy}^{-1} \Lambda_{yx}
   \right)\mathbf{x}\\
   &+\frac 1 2 \mu_x^T \Lambda_{xy} \Lambda_{yy}^{-1} \Lambda_{yx} \mu_x-\frac  1 2\mu_x^T \Lambda_{xx} \mu_x\\
   &= -\frac 1 2(\mathbf{x} - \mu_x)^T (\Lambda_{xx} - \Lambda_{xy}\Lambda_{yy}^{-1} \Lambda_{yx})(\mathbf{x}-\mu_x)
\end{aligned}
$$

根据[分块矩阵逆矩阵](https://en.wikipedia.org/wiki/Block_matrix)的形式

$$
\Sigma^{-1} = \left[
\begin{aligned}
\Sigma_{xx}^{-1}+\Sigma_{xx}^{-1} \Sigma_{xy} M \Sigma_{yx} \Sigma_{xx}^{-1} & \quad - \Sigma_{xx}^{-1} \Sigma_{xy}M\\
-M\Sigma_{yx} \Sigma_{xx}^{-1}  & \quad M
\end{aligned}
\right]
$$

其中 \\(M = (\Sigma_{yy} - \Sigma_{yx} \Sigma_{xx}^{-1} \Sigma_{xy})^{-1}\\)

可以得出

$$
\begin{aligned}
&\Lambda_{xx} - \Lambda_{xy}\Lambda_{yy}^{-1} \Lambda_{yx}\\
&= \Sigma_{xx}^{-1}+\Sigma_{xx}^{-1} \Sigma_{xy} M \Sigma_{yx} \Sigma_{xx}^{-1} - \Sigma_{xx}^{-1} \Sigma_{xy}M M^{-1}M\Sigma_{yx} \Sigma_{xx}^{-1} \\
&=\Sigma_{xx}^{-1}
\end{aligned}
$$

于是

$$
p(\mathbf{x})=\frac 1 {\sqrt{(2\pi)^{d-k} |\Sigma| |\Lambda_{yy}|}}\exp\left(-\frac 1 2(\mathbf{x} - \mu_x)^T \Sigma_{xx}^{-1}(\mathbf{x} - \mu_x)\right)
$$

另一方面，[分块矩阵的行列式](http://djalil.chafai.net/blog/2012/10/14/determinant-of-block-matrices/)为

$$
|\Sigma| = |\Sigma_{xx}||M|^{-1}
$$

再根据方阵乘积的行列式关系

$$
|AB| = |A||B|
$$

可见

$$
\begin{aligned}\\
|\Sigma||\Lambda_{yy}| &= |\Sigma_{xx}||M|^{-1}|M| = |\Sigma_{xx}|
\end{aligned}
$$

那么我们看到

$$
p(\mathbf{x})=\frac 1 {\sqrt{(2\pi)^{d-k}  |\Sigma_{xx}|}}\exp\left(-\frac 1 2(\mathbf{x} - \mu_x)^T \Sigma_{xx}^{-1}(\mathbf{x} - \mu_x)\right)
$$

这一结果就表明，当联合分布为高斯分布时，那么其中一部分随机变量的边缘分布也为高斯分布，并且其均值和协方差矩阵就是原分布对应分量的均值和协方差矩阵。



### 条件分布——二维情况

现在回到二维分布，获得了 \\(p(x)\\) ，我们便可以利用贝叶斯定理，求得条件概率分布

$$
p(y|x) = \frac{p(x, y)}{p(x)} =\frac{\frac 1 {2\pi \sqrt{|\Sigma|}}\exp(C(x) +C(y)+C) }
{\frac 1 {\sqrt{2\lambda_{22}\pi|\Sigma|}}\exp(C(x) + C + C_2(x))} \\
=\sqrt{\frac{\lambda_{22}}{2\pi}}\exp(C(y)-C_2(x))
$$

而根据前面对 \\(C(y)\\) 配平方的推导

$$C(y) = -z^2 + C_2(x)$$

所以我们看到

$$
\begin{aligned}
p(y\mid x) &= \sqrt{\frac {\lambda_{22}}{2\pi}}\exp\left(
-\frac{\lambda_{22}}{2}\left(y - \frac{\lambda_{22}\mu_2 + \lambda_{12}x -\lambda_{12}\mu_1}{\lambda_{22}}\right)^2
  \right)\\
  &=\frac{1}{\sqrt{2\pi}\sqrt{\frac{1}{\lambda_{22}}}} \exp\left(
-\frac{\left(y - \frac{\lambda_{22}\mu_2 + \lambda_{12}x -\lambda_{12}\mu_1}{\lambda_{22}}\right)^2}{2 \frac{1}{\lambda_{22}}}
    \right)
\end{aligned}
$$

也就是说，*y* 关于 *x* 的条件概率服从均值为 \\(\frac{\lambda_{22}\mu_2 + \lambda_{12}x -\lambda_{12}\mu_1}{\lambda_{22}}\\)，方差为 \\(\frac{1}{\lambda_{22}}\\) 的高斯分布。

### 条件分布——高维情况

然后再看看高维情况

$$
\begin{aligned}

p(\mathbf{y}\mid \mathbf{x}) &= \frac{p(\mathbf{x}, \mathbf{y})}{p(\mathbf{x})}\\
&= \frac{
  \frac{1}{\sqrt{(2\pi)^d |\Sigma|}}\exp(C(\mathbf{x})+C(\mathbf{y})+C)
  }{
     \frac 1 {\sqrt{(2\pi)^{d-k} |\Sigma| |\Lambda_{yy}|}}\exp(C(\mathbf{x})+C_2(\mathbf{x})+C)}\\
     &=\sqrt{\frac{|\Lambda_{yy}|}{(2\pi)^k}}\exp(C(\mathbf{y}) - C_2(\mathbf{x}))\\
     &=\sqrt{\frac{|\Lambda_{yy}|}{(2\pi)^k}}\exp\left(
-\frac 1 2 \mathbf{t}^T \Lambda_{yy} \mathbf{t}
       \right)\\
       &=\frac{1}{(2\pi)^k |\Lambda_{yy}^{-1}|}\exp\left(
  -\frac 1 2 ( \mathbf{y} -\Lambda_{yy}^{-1}\mathbf{m})^T (\Lambda_{yy}^{-1})^{-1}(\mathbf{y} -\Lambda_{yy}^{-1}\mathbf{m}))
         \right)
  \end{aligned}
$$

也就是说，条件分布 \\(p(\mathbf{y}\mid\mathbf{x})\\) 是以 \\(\Lambda_{yy}^{-1}\\) 为协方差矩阵，\\(\Lambda_{yy}^{-1} \mathbf{m}\\) 为均值的高斯分布，其中 \\(\mathbf{m}^T = \mu_{y}^T \Lambda_{yy} - \mu_x^T \Lambda_{xy} + \mathbf{x}^T \Lambda_{xy}\\)。

这里我们总结一下，如果两个随机向量 \\(\mathbf{x},  \mathbf{y}\\) 的联合分布 \\(p(\mathbf{x},\mathbf{y})\\) 为多元高斯分布，并假设它的均值和协方差矩阵分别具有下面的分块形式

$$
\mu = \left(\begin{aligned}
\mu_x\\
\mu_y
\end{aligned}\right)
$$

$$
\Sigma = \left[
\begin{aligned}
\Sigma_{xx} &\quad\Sigma_{xy}\\
\Sigma_{yx} &\quad\Sigma_{yy}
\end{aligned}
\right]
$$

并且为了表达上的方便，还定义了协方差矩阵的逆矩阵

$$
\Sigma^{-1} = \left[
\begin{aligned}
\Lambda_{xx} &\quad\Lambda_{xy}\\
\Lambda_{yx} &\quad\Lambda_{yy}
\end{aligned}
\right]
$$

当然协方差矩阵和逆矩阵的分块矩阵存在下述关系

$$
\left[
\begin{aligned}
\Lambda_{xx} &\quad\Lambda_{xy}\\
\Lambda_{yx} &\quad\Lambda_{yy}
\end{aligned}
\right] = \left[
\begin{aligned}
\Sigma_{xx}^{-1}+\Sigma_{xx}^{-1} \Sigma_{xy} M \Sigma_{yx} \Sigma_{xx}^{-1} & \quad - \Sigma_{xx}^{-1} \Sigma_{xy}M\\
-M\Sigma_{yx} \Sigma_{xx}^{-1}  & \quad M
\end{aligned}
\right]
$$

其中 \\(M = (\Sigma_{yy} - \Sigma_{yx} \Sigma_{xx}^{-1} \Sigma_{xy})^{-1}\\)。有了上面这些知识，我们便可写出边缘分布的形式

$$
\mathbf{x} \sim N(\mu_x, \Sigma_{xx})
$$

以及条件分布的形式

$$
\mathbf{y}\mid \mathbf{x} \sim N(\Lambda_{yy}^{-1} \mathbf{m},\,\Lambda_{yy}^{-1})
$$

当然 \\(\mathbf{y}\\) 和 \\(\mathbf{x}\mid\mathbf{x}\\) 的分布形式只需作稍微的参数调换，本质都一样，这里不再赘述。

### 从条件分布到联合分布

在前面，我们将随机向量拆分成了两部分，推导了它们的联合分布为多元高斯分布情况下的边缘分布和条件分布公式，发现仍然是高斯分布，并且均值和协方差矩阵都能通过联合分布的相关参数计算。

接下来，我们将这个过程反过来，通过两个随机向量的条件分布以及其中一个向量的边缘分布，来计算它们的联合分布，以及另一个随机向量的边缘分布。

我们假设随机向量 \\(\mathbf{x}\\) 的维度为 *s*， 服从高斯分布 \\(N(\mu_x, \Sigma_{x})\\)，随机向量 \\(\mathbf{y}\\) 的维度为 *k*，并且关于 \\(\mathbf{x}\\) 的条件分布服从高斯分布 \\(N(\mu_{y\mid x}, \Sigma_{y\mid x})\\)。考虑到前面我们在根据联合分布推导条件分布时得到的结果，\\(p(\mathbf{y}\mid\mathbf{x})\\) 的均值

$$
\Lambda_{yy}^{-1}\mathbf{m} = \Lambda_{yy}^{-1}\Lambda_{yy}^T \mu_{y}-  \Lambda_{yy}^{-1}\Lambda_{xy}^T \mu_x +\Lambda_{yy}^{-1}\Lambda_{xy}^T \mathbf{x}
$$

也就是说条件分布的均值为 \\(\mathbf{x}\\) 的线性函数，于是我们进一步假设

$$
\mu_{y \mid x} = A \mathbf{x} + b
$$

其中 *A* 为 *k x s* 矩阵。利用这一关系，通过下面的公式计算联合分布

$$
\begin{aligned}
p(\mathbf{x}, \mathbf{y}) &= p(\mathbf{y}\mid \mathbf{x})p(\mathbf{x}) \\
&=\frac 1 {\sqrt{(2\pi)^k |\Sigma_{y\mid x}|}}\exp\left(
-\frac 1 2 (\mathbf{y} - A \mathbf{x} - b)^T\Sigma_{y\mid x}^{-1}(\mathbf{y} - A \mathbf{x} - b)
  \right)
  \\
  &\cdot \frac 1 {\sqrt{(2\pi)^s |\Sigma_{x}|}}\exp\left(
  -\frac 1 2 (\mathbf{x} - \mu_{x})^T\Sigma_{x}^{-1}(\mathbf{x} - \mu_{x})
    \right)
\end{aligned}
$$

为了方便起见，现在我们单独考虑上式的指数项，由于指数函数的乘法等于底数不变，指数项相加。

$$
\begin{aligned}
&-\frac 1 2 (\mathbf{y} -A \mathbf{x} - b)^T\Sigma_{y\mid x}^{-1}(\mathbf{y} - A \mathbf{x} - b)  -\frac 1 2 (\mathbf{x} - \mu_{x})^T\Sigma_{x}^{-1}(\mathbf{x} - \mu_{x})\\
&=-\frac 1 2 \left(
\mathbf{x}^T\Sigma_{x}^{-1}\mathbf{x}
-\mathbf{x}^T\Sigma_{x}^{-1}\mu_x
-\mu_x^T\Sigma_{x}^{-1}\mathbf{x}
+\mu_x^T\Sigma_{x}^{-1} \mu_x+ \mathbf{y}^T\Sigma_{y\mid x}^{-1}\mathbf{y}
-\mathbf{y}^T\Sigma_{y\mid x}^{-1}(A \mathbf{x} + b)
-(A \mathbf{x} + b)^T\Sigma_{y\mid x}^{-1}\mathbf{y}
+(A \mathbf{x} + b)^T\Sigma_{y\mid x}^{-1} (A \mathbf{x} + b)
  \right)\\
  &=-\frac 1 2 \left(
\mathbf{x}^T (\Sigma_x^{-1}+A^T \Sigma_{y\mid x}^{-1}A) \mathbf{x}- \mathbf{x}^T (\Sigma_x^{-1}\mu_x - A^T \Sigma_{y\mid x}^{-1}b)- (\mu_x^T \Sigma_x^{-1}-b^T \Sigma_{y\mid x}^{-1}A)\mathbf{x}- \mathbf{y}^T \Sigma_{y\mid x}^{-1}A \mathbf{x}
-\mathbf{x}^T  A^T \Sigma_{y\mid x}^{-1}\mathbf{y}
-\mathbf{y}^T \Sigma_{y\mid x}^{-1} b
-b^T \Sigma_{y\mid x}^{-1} \mathbf{y}
+\mathbf{y}^T\Sigma_{y\mid x}^{-1}\mathbf{y}
+\mu_x^T\Sigma_{x}^{-1} \mu_x
+b^T\Sigma_{y\mid x}^{-1} b
    \right)\\
    &=-\frac 1 2\left(
      \mathbf{x}^T (\Sigma_x^{-1}+A^T \Sigma_{y\mid x}^{-1}A) \mathbf{x}
-\mathbf{x}^T  A^T \Sigma_{y\mid x}^{-1}\mathbf{y}
-\mathbf{y}^T \Sigma_{y\mid x}^{-1}A \mathbf{x}
 +\mathbf{y}^T\Sigma_{y\mid x}^{-1}\mathbf{y}
-2\mathbf{x}^T \Sigma_x^{-1}\mu_x+ 2\mathbf{x}^T A^T \Sigma_{y\mid x}^{-1} b - 2\mathbf{y}^T \Sigma_{y\mid x}^{-1} b
-\frac 1 2 \mu_x^T\Sigma_{x}^{-1} \mu_x
-\frac 1 2 b^T\Sigma_{y\mid x}^{-1} b
      \right)\\
    &=- \frac 1 2 (\mathbf{x}^T \quad \mathbf{y}^T)
    \left[
    \begin{aligned}
    \Sigma_x^{-1}+A^T \Sigma_{y\mid x}^{-1}A & \quad -A^T \Sigma_{y\mid x}^{-1}\\ -\Sigma_{y\mid x}^{-1}A&\quad\Sigma_{y\mid x}^{-1}
    \end{aligned}
            \right]
            \left(
            \begin{aligned}
            \mathbf{x}\\\mathbf{y}
            \end{aligned}
                    \right)\\
                    &+(\mathbf{x}^T\quad \mathbf{y}^T)
                    \left(
            \begin{aligned}
            \Sigma_x^{-1}\mu_x - A^T \Sigma_{y\mid x}^{-1} b\\
            \Sigma_{y\mid x}^{-1} b
            \end{aligned}
                    \right)
                    -\frac 1 2 \mu_x^T\Sigma_{x}^{-1} \mu_x
                    -\frac 1 2 b^T\Sigma_{y\mid x}^{-1} b\\
                    &= -\frac 1 2 \mathbf{z}^T R \mathbf{z} + \mathbf{z}^T S + C
\end{aligned}
$$

公式的最后我们定义了一些符号以简化表达，然后再使用待定系数法对上式结果配平方

$$
\begin{aligned}
-\frac 1 2 \mathbf{z}^T R \mathbf{z} + \mathbf{z}^T S + C
&= -\frac 1 2 (\mathbf{z} - a)^T R (\mathbf{z} - a) + e\\
&= -\frac 1 2 \mathbf{z}^T R \mathbf{z} + \mathbf{z}^T Ra - \frac 1 2 a^T R a + e
\end{aligned}
$$

根据上式中的对应项相等，我们可以得到待定系数的值

$$
a = R^{-1} S\\
e = \frac 1 2 S^T R^{-1} S + C
$$

利用分块矩阵的逆矩阵公式

$$
\left[
\begin{aligned}
A \quad & B\\
C \quad & D
\end{aligned}
\right]=
\left[
\begin{aligned}
M_z^{-1} \quad & - M_z B D ^{-1}\\
-D^{-1} CM_z \quad & D^{-1} + D^{-1} C M^{-1} B D^{-1}
\end{aligned}
\right]
$$

其中 \\(M = A - B D^{-1} C\\)。可以得到 \\(R^{-1}\\) 如下

$$
R^{-1} = \left[
\begin{aligned}
\Sigma_x \quad & \Sigma_x A^T\\
A \Sigma_x \quad & \Sigma_{y\mid x} + A \Sigma_x A^{T}
\end{aligned}
\right]
$$

然后再具体计算 *a e* 得到

$$
a = \left(
\begin{aligned}
&\mu_x \\ A \mu_x &+ b
\end{aligned}
\right)\\
e= 0 \qquad \qquad \,  \, \,
$$

所以我们看到，联合分布的指数项其实是一个完全平方项，所以

$$
p(\mathbf{x}, \mathbf{y}) = \frac 1 {\sqrt{(2\pi)^{k+s} |\Sigma_{y\mid x}||\Sigma_x|}}\exp\left(-\frac 1 2(\mathbf{z} - a)^T R (\mathbf{z} - a)\right)
$$

根据逆分块矩阵的行列式表达式

$$
\left|
\begin{aligned}
A&\quad B\\
C&\quad D
\end{aligned}
  \right|^{-1} = |D - C A^{-1} B||A|
$$

可以得到 \\(R^{-1}\\) 的行列式

$$
R^{-1} = |\Sigma_{y\mid x} + A \Sigma_x A^T  - A \Sigma_x \Sigma_x^{-1} \Sigma_x A^T ||\Sigma_x| = |\Sigma_{y\mid x}||\Sigma_x|
$$

所以联合分布的表达式就为

$$
p(\mathbf{x}, \mathbf{y}) = \frac 1 {\sqrt{(2\pi)^{k+s} |R^{-1}|}}\exp\left(-\frac 1 2(\mathbf{z} - a)^T R (\mathbf{z} - a)\right)
$$

也就是说 \\(\mathbf{x}, \mathbf{y}\\) 的联合分布是一个均值为 \\(a\\)，协方差矩阵为 \\(R^{-1}\\) 的多元高斯分布。再结合前面通过联合分布求边缘分布和条件分布的讨论，可知随机向量 \\(\mathbf{y}\\) 服从高斯分布，以及 \\(\mathbf{x}\\) 关于 \\(\mathbf{y}\\) 的条件分布也是高斯分布。

参考：

C.M. Bishop: 模式识别与机器学习

---
码字不易，您的支持是我最大的动力，感谢打赏~

![](/resources/alipay.jpg)