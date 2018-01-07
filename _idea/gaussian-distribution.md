## 条件高斯分布

均值为 $$\mu$$ ，协方差矩阵为 $$\Sigma$$ 的多元高斯分布为

$$
p(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} \exp\left(
-\frac 1 2 (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)
\right)
$$

为了讨论简单起见，我们先考虑变量为 $$x, y$$ 的二元高斯分布，此时均值和协方差矩阵分别为

$$
\mu^T = (\mu_1\quad \mu_2)^T\\
\Sigma =
\left[
\begin{aligned}
\sigma_{11} &\quad \sigma_{12}\\\sigma_{21}&\quad\sigma_{22}
\end{aligned}
\right]
$$

其中 $$\sigma_{12} = \sigma_{21}$$。为了表达方便，这里设协方差矩阵的逆矩阵为

$$
\Sigma^{-1} =
\left[
\begin{aligned}
\lambda_{11} &\quad \lambda_{12}\\\lambda_{21}&\quad\lambda_{22}
\end{aligned}
\right]
$$

其中 $$\lambda_{12} = \lambda_{21}$$。于是高斯分布的表达式就为

$$
p(x, y) =  \frac{1}{2\pi \sqrt{|\Sigma|}} \exp\left(-\frac 1 2
\lambda_{11}(x-\mu_1)^2 -\frac 1 2 \lambda_{22} (y-\mu_2)^2 + \lambda_{12}(x -\mu_1)(y-\mu_2
)
\right)
$$

### 边缘分布

为了计算多元高斯分布的边缘分布，我们需要对其余变量在全空间上积分。也就是说

$$
p(x) = \int_{-\infty}^{\infty} p(x, y)\mathrm{d}y
$$

为了解析地求得这个积分，我们先把联合分布 $$p(x, y)$$ 的指数项展开
$$
\begin{aligned}
&-\frac 1 2
\left(\lambda_{11}(x-\mu_1)^2 + \lambda_{22} (y-\mu_2)^2 -2 \lambda_{12}(x -\mu_1)(y-\mu_2)\right)\\
&= -\frac 1 2
\left(
\lambda_{11} x^2 -2\lambda_{11}x\mu_1 + \lambda_{11}\mu_1^2
+ \lambda_{22}y^2 -2\lambda_{22} y\mu_2 +\lambda_{22}\mu_2^2 \\
-2\lambda_{12}xy + 2\lambda_{12} x\mu_2 + 2\lambda_{12} y\mu_1 - 2\lambda_{12}\mu_1 \mu_2
\right)\\
&=(-\frac 1 2 \lambda_{11}x^2 + \lambda_{11}\mu_1x -\lambda_{12}\mu_2 x)
+(-\frac 1 2 \lambda_{22}y^2 + \lambda_{22}\mu_2 y + \lambda_{12} x y - \lambda_{12} \mu_1  y)\\
&-\frac 1 2(\lambda_{11}\mu_1^2+\lambda_{22}\mu_2^2  -2 \lambda_{12}\mu_1 \mu_2)\\
&= C(x) + C(y) + C
\end{aligned}

$$

为了简化表达，我们令 $$C(x), C(y), C$$ 来分别表达含有 *x*，*y* 的项和常数项。
由于是对 *y* 的积分，所以与 *y* 无关的项可以提到积分符号外面去，即得到

$$
p(x) = \frac{1}{2\pi \sqrt{|\Sigma|}} \exp(C(x)+C) \int_{-\infty}^{\infty} e^{C(y)} \mathrm{d}y
$$

为了计算[高斯积分](https://en.wikipedia.org/wiki/Gaussian_integral)，我们先对 $$C(y)$$ 进行配平方

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

同样，为了简化表达，上式中令 $$z = \sqrt{\frac{\lambda_{22}}{2}} \left(y-\frac{\lambda_{22}\mu_2 +\lambda_{12}x -\lambda_{12}\mu_1}{\lambda_{22}}\right)$$，第二项又与 *y* 无关。
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
&= -\frac{\lambda_{11} \lambda_{22} - \lambda_{12}^2}{2\lambda_{22}}x^2
+ \frac{\lambda_{11}\lambda_{22} - \lambda_{12}^2}{\lambda_{22}}\mu_1 x + C' +C\\
&=- \frac{\lambda_{11} \lambda_{22} - \lambda_{12}^2}{2\lambda_{22}}(x^2 -2 \mu_1 x) +C' +C\\
&= - \frac{\lambda_{11} \lambda_{22} - \lambda_{12}^2}{2\lambda_{22}}(x-\mu_1)^2
+ \frac{\lambda_{11} \lambda_{22} - \lambda_{12}^2}{2\lambda_{22}}\mu_1^2 +C'+C\\
&= - \frac{\lambda_{11} \lambda_{22} - \lambda_{12}^2}{2\lambda_{22}}(x-\mu_1)^2
+ C'' +C'+C
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

其中 $$D = C''+C'+C$$，如果仔细算一算，会发现其实 $$D = 0$$。所以我们看到

$$
p(x) = \frac{1}{\sqrt{2\pi} (\lambda_{22}|\Sigma|)^{\frac 1 2}} \exp(-\frac{(x - \mu_1)^2}{2\lambda_{22}|\Sigma|} )
$$

也就是说，*x* 的分布为以 $$\mu_1$$ 为均值，$$\lambda_{22}|\Sigma|$$ 为方差的高斯分布，除此之外它还有个系数 $$\frac{e^D}{\sqrt{2}}$$。

我们仔细考察一下边缘分布的均值与方差，容易发现，其均值实际上刚好等于联合分布下 *x* 分量的均值。然后我们再简单计算一下协方差矩阵的逆矩阵

$$
\Sigma^{-1} =\frac{1}{|\Sigma|} \left[
\begin{aligned}
\sigma_{22} &\quad -\sigma_{12}\\-\sigma_{21}&\quad\sigma_{11}
\end{aligned}
\right]
$$

于是可以得到 $$\lambda_{22} = \frac{\sigma_{11}}{|\Sigma|}$$，那么边缘分布的方差其实就等于 $$\sigma_{11}$$，而这其实就是原联合分布下 *x* 分量的方差。最终可得

$$
p(x) = \frac{1}{\sqrt{2\pi  \sigma_{11}} } \exp\left(-\frac{(x - \mu_1)^2}{2\sigma_{11}} \right)
$$

从上面的分析，我们看到一个事实：对于二元高斯分布，其边缘分布是一个高斯分布，并且它的均值和方差就是原联合分布对应分量的均值和方差。

这是一个很强的结论，它让我们可以不必通过计算，直接通过联合分布的表达式写出边缘分布的表达式，下面我们试着将其推广到更高阶的情形。

设 d 维随机向量

$$
\mathbf{z} = \left(
\begin{aligned}
\mathbf{x} \\ \mathbf{y}
\end{aligned}
  \right)
$$

其中 $$\mathbf{x},\mathbf{y}$$ 都是向量，它们的维数和等于 d，并且 $$\mathbf{z}$$ 的分布为高斯分布，即

$$
p(\mathbf{z}) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} \exp\left(
-\frac 1 2 (\mathbf{z} - \mu)^T \Sigma^{-1} (\mathbf{z} - \mu)
\right)
$$

根据 $$\mathbf{x},\mathbf{y}$$ 对 $$\mathbf{z}$$ 的分割，将均值和协方差矩阵按如下分割

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

由于协方差矩阵的对称性，我们知道 $$\Sigma_{xy} = \Sigma_{yx}^T$$。为了表达的方便，我们再假设协方差矩阵的逆矩阵为

$$
\Sigma^{-1} = \left[
\begin{aligned}
\Lambda_{xx}\quad & \Lambda_{xy} \\
\Lambda_{yx}\quad & \Lambda_{yy}
\end{aligned}
  \right]
$$

同样 $$\Lambda_{xy} = \Lambda_{yx}$$。利用上述分解，我们可以将联合分布的指数项写成

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
&+\mathbf{x}^T \Lambda_{xy} \mathbf{y}-\mu_x^T \Lambda_{xy} \mathbf{y}-\mu_y^T \Lambda_{xy} \mathbf{x} + \mu_x^T \Lambda_{xy} \mu_y \\
&= \left(
-\frac 1 2 \mathbf{x}^T \Lambda_{xx} \mathbf{x} + (\mu_x^T \Lambda_{xx} - \mu_y^T \Lambda_{xy})\mathbf{x}
  \right)
  +\left(
-\frac 1 2 \mathbf{y}^T \Lambda_{yy} \mathbf{y} +(\mu_{y}^T \Lambda_{yy} - \mu_x^T \Lambda_{xy} - \mathbf{x}^T \Lambda_{xy})\mathbf{y}
    \right)\\
    &+\left(
 -\frac 1 2 \mu_x^T \Lambda_{xx} \mu_x-\frac 1 2 \mu_y^T \Lambda_{yy} \mu_y + \mu_x^T \Lambda_{xy} \mu_y
      \right)\\
      &=C(\mathbf{x}) +C(\mathbf{y})+C
\end{aligned}
$$

这里为了方便表达，我们仍然使用 $$C(\mathbf{x}), C(\mathbf{y}),C$$ 来分别表示与 $$\mathbf{x}, \mathbf{y}$$ 相关的项和无关项。

现在我们要求边缘分布 $$p(\mathbf{x})$$，这需要在全空间上对 $$\mathbf{y}$$ 积分

$$
\begin{aligned}
p(\mathbf{x}) &= \int p(\mathbf{x}, \mathbf{y}) \mathrm{d}\mathbf{y}\\
&= \frac 1 {\sqrt{(2\pi)^d |\Sigma|}}e^{C(\mathbf{x})+C}\int e^{C(\mathbf{y})}\mathrm{d}\mathbf{y}
\end{aligned}
$$

对 $$C(\mathbf{y})$$ 配平方

$$
\begin{aligned}
C(\mathbf{y}) &= -\frac 1 2 \mathbf{y}^T \Lambda_{yy} \mathbf{y} +(\mu_{y}^T \Lambda_{yy} - \mu_x^T \Lambda_{xy} - \mathbf{x}^T \Lambda_{xy})\mathbf{y}\\
&=?
\end{aligned}
$$

### 条件分布

获得了 $$p(x)$$ ，我们便可以利用贝叶斯定理，求得条件概率分布

$$
p(y|x) = \frac{p(x, y)}{p(x)} =\frac{\frac 1 {2\pi \sqrt{|\Sigma|}}\exp(C(x) +C(y)+C) }
{\frac 1 {2\sqrt{\lambda_{22}\pi|\Sigma|}}\exp(C(x) + C + C_2(x))} \\
=\sqrt{\frac{\lambda_{22}}{\pi}}\exp(C(y)-C_2(x))
$$

而根据前面对 $$C(y)$$ 配平方的推导

$$C(y) = -z^2 + C_2(x)$$

所以我们看到

$$
\begin{aligned}
p(y\mid x) &= \sqrt{\frac {\lambda_{22}}{\pi}}\exp\left(
-\lambda_{22}\left(y - \frac{\lambda_{22}\mu_2 + \lambda_{12}x -\lambda_{12}\mu_1}{\lambda_{22}}\right)^2
  \right)\\
  &=\frac{1}{\sqrt{2\pi}\sqrt{\frac{1}{2\lambda_{22}}}} \exp\left(
-\frac{\left(y - \frac{\lambda_{22}\mu_2 + \lambda_{12}x -\lambda_{12}\mu_1}{\lambda_{22}}\right)^2}{2 \frac{1}{2\lambda_{22}}}
    \right)
\end{aligned}
$$

也就是说，*y* 关于 *x* 的条件概率服从均值为 $\frac{\lambda_{22}\mu_2 + \lambda_{12}x -\lambda_{12}\mu_1}{\lambda_{22}}$$，方差为 $\frac{1}{2\lambda_{22}}$$ 的高斯分布。

end



end



end



end
