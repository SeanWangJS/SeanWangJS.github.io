---
layout: post
title: 理解主成分分析（1）
---

如果样本点具有很高维的特征，那么我们就必须要使用很长的数组去存储，但是如果这些特征中有许多都是不必要的，那么这不仅浪费了存储空间，而且也为训练过程带来了额外的计算量。例如在手写数字识别中，需要将图片分割成 n by n 的矩阵，令有像素的块为 1，没有像素的块为 0，然后再用一个 n*n 容量的数组去存储一副图片的特征。

![](/resources/2017-12-21-principal-components-analysis/digits.png)
（每个图片可以使用矩阵来分割，并且存储成数组）

但是这样的数组大部分的分量都是 0，而这些 0 并不能作为某个数字的特征。所以很多时候我们需要对拿到的数据进行处理，提取出具有代表性的特征，才能更好地加入训练模型之中。

主成分分析便是一种有效的数据降维处理方法，为了对问题进行说明，我们先用具有两个维度的样本集作为例子。

设有特征集

$$
S = \{x_i\mid i=1,2,,,n\}
$$

其中每个点 $$x_i$$ 含有两个分量。那么根据点集的分布情况，可以发现，在某些方向上，点的散布性更强，而更强的散布性则意味着更好的代表性。

![](/resources/2017-12-21-principal-components-analysis/distribution.png)
（散点的分布在一个方向上有很好的散布性，而在另一个方向上的散布性稍差）

为了衡量这种散布性，我们考虑一条过原点的直线

$$
\omega^T x = 0
$$

其中 $$\omega^T = [\omega_1 \quad \omega_2]$$，并且规定 $$\|\omega\| =1$$。然后定义函数

$$
f(x) = \omega^T x
$$

可以证明这是点 $$x$$ 到直线 $$\omega^T x = 0$$ 的距离。于是我们可以使用这条直线作为基线，将特征映射成点到直线的距离，然后利用距离的散布作为特征的散布。

若定义

$$
Y = \{y_i = \omega^T x_i\mid x_i \in S\}

$$

然后利用方差作为数据的散布衡量，那么我们的目标其实就是寻找 $$\omega$$ 使得

$$
Var(Y)
$$

具有最大值。

为了一般性讨论，我们考虑更高维的数据，假设特征维度为 *d* ，那么

$$
y_i = \omega_1 x_{i1} +  \omega_2 x_{i2}  + ... +  \omega_d x_{id}
$$

并且有

$$
Y = \left[
\begin{aligned}
y_1 \\y_2\\...\\y_n
\end{aligned}
\right]

=\omega_1\left[
\begin{aligned}
x_{11} \\x_{21}\\...\\x_{n1}
\end{aligned}
\right]
+
\omega_2\left[
\begin{aligned}
x_{12} \\x_{22}\\...\\x_{n2}
\end{aligned}
\right]
+ ...
 +
 \omega_d\left[
 \begin{aligned}
 x_{1d} \\x_{2d}\\...\\x_{nd}
 \end{aligned}
 \right]
$$

如果再定义

$$
X_i = \left[
\begin{aligned}
x_{1i} \\x_{2i}\\...\\x_{ni}
\end{aligned}
\right]
$$

那么就有

$$
Y = \sum_{i=1}^d \omega_i X_i
$$

我们在这里将所有特征点作为一个矩阵来考虑，即

$$
X = [X_1 \quad X_2 \quad ...\quad X_d]
$$

然后假设其协方差矩阵为

$$
\Sigma = \left[
\begin{aligned}
\sigma_{11}\quad&\sigma_{12}\quad..&\sigma_{1d} \\
\sigma_{21}\quad&\sigma_{22}\quad..&\sigma_{2d}\\
..\quad&..\,\,\,\quad..&..\\
\sigma_{d1}\quad&\sigma_{d2}\quad..&\sigma_{dd}\\
\end{aligned}
\right]
$$

其中 $$\sigma_{ij} = Cov(X_1, C_2)$$。下面我们将证明 $$Var(Y)$$ 的最大值为 $$\Sigma$$ 的最大特征值。

根据方差的性质

$$
Var(Y) = \sum_{i=1}^d\sum_{j=1}^d \omega_i \omega_j \sigma_{ij} = \omega^T \Sigma \omega
$$

容易知道 $$\Sigma$$ 为实对称矩阵，于是它拥有一组可以作为单位正交基的特征向量

$$
e_1,e_2...e_d
$$

满足

$$
e_i^T e_j = \left\{
\begin{aligned}
  1\quad i = j\\0\quad i \ne  j
\end{aligned}\right.
$$

以及相对应的特征值

$$
\lambda_1, \lambda_2 ... \lambda_d
$$

并不失一般性的假设

$$
\lambda_1 \ge \lambda_2 \ge ... \ge \lambda_d
$$


利用这组基向量，我们可以将 $$\omega$$ 分解成

$$
\omega = \sum_{j=1}^d \alpha_j e_j
$$

那么

$$
Var(Y) = \sum_{i=1}^d \sum_{j=1}^d \alpha_i \alpha_j e_i^T \Sigma e_j = \sum_{i=1}^d \alpha_i^2 \lambda_i \le \lambda_1\sum_{i=1}^n\alpha_i^2
$$

其中 $$\lambda_1$$ 是最大的特征值，上述公式取等号的条件是，与 $$\lambda_1$$ 相对应的系数 $$\alpha_1$$ 为 1，其他的系数为 0。再考虑到约束

$$
\|\omega\|^2=\sum_{i=1}^d \sum_{j=1}^d \alpha_i \alpha_j e_i e_j = \sum_{i=1}^n \alpha_i^2 =1
$$

于是可以得出结论

$$
Var(Y) \le  \lambda_1
$$

并且当 $$\omega = e_1$$ 时，等号成立，对应的映射值就为

$$
Y = \sum_{i=1}^d e_{1i} X_i
$$

这里的 $$Y$$ 就是对 $$X$$ 的一种表征，它只有一个维度，并且与最大的特征值相关，所以称之为 $$X$$ 的第一主成分。

但是显然，虽然第一主成分能够一定程度上代表原始特征，却失去了原始特征的很多信息，为了尽量补全信息，下面我们将考虑更多的次要成分。

在我们计算第一主成分的时候，考虑的是用一条直线（在高维情况下为超平面）作为基线，然后将特征点投影上去，用距离作为特征点的代表，并且这条直线使得距离有很好的散布性。同样的道理，计算第二个主成分，也是要找到一条直线或者超平面来做投影，但是它必须要和计算第一主成分时的基线正交，否则得到的第二主成分将有一部分含有第一主成分的信息，这显然是不合理的。于是计算第二主成分的系数向量 $$\omega^{(2)}$$ 必须满足条件

$$
\omega^{(2)} \cdot e_1  = 0
$$

以及当然

$$
\|\omega^{(2)}\| = 1
$$

然后我们将证明，满足上述条件的 $$\omega^{(2)}$$ 必然使

$$
Y_2 = \sum_{i=1}^d \omega^{(2)}_i X_i
$$

最大为 $$\Sigma$$ 第二大的特征值 $$\lambda_2$$，并且 $$\omega^{(2)} = e_2$$。

首先对 $$\omega^{(2)}$$ 进行分解

$$
\omega^{(2)} = \sum_{i=1}^d \alpha_i^{(2)} e_i
$$

考虑到 $$\omega^{(2)}$$ 与 $$e_1$$ 正交，所以

$$
\begin{aligned}
\omega^{(2)} \cdot e_1 &= 0\\
\sum_{i=1}^d \alpha_i^{(2)} e_i \cdot e_1 & = 0\\
\alpha_1^{(2)} &= 0
\end{aligned}
$$

也就是说，第一个方向上的分量为 0。然后根据方差的性质

$$
\begin{aligned}
  Var(Y_2) &= \sum_{i=1}^d \sum_{j=1}^d \alpha_i^{(2)} \alpha_j^{(2)} e_i^T \Sigma e_j \\&= \sum_{i=1}^d \sum_{j=1}^d \alpha_i^{(2)} \alpha_j^{(2)} e_i^T \lambda_j e_j\\
  &= \sum_{i=1}^d \alpha_i^{(2)} \alpha_i^{(2)} \lambda_i\\
  &= \sum_{i=2}^d \alpha_i^{(2)} \alpha_i^{(2)} \lambda_i\\
  &\le \lambda_2 \sum_{i=2}^d \alpha_i^{(2)} \alpha_i^{(2)}
\end{aligned}
$$

同样，根据 $$\|\omega^{(2)}\|$$，可得

$$
\sum_{i=1}^d \alpha_i^{(2)}  \alpha_i^{(2)} =0
$$

于是有

$$
Var(Y_2) \le \lambda_2
$$

其中取等号的条件是，不等式中 $$\alpha_2^{(2)} = 1$$， 其余系数为 0。于是我们就证明了，当 $$\omega_{(2)} = e_2$$ 时，$$Var(Y_2)$$ 取得最大值 $$\lambda_2$$。

基于同样的方法，我们还能寻找第三主成分，第四主成分等等。将这些主成分 $$Y_1, Y_2, Y_3 ...$$ 按列合并，形成新的特征矩阵

$$
Y = [Y_1 \quad Y_2 \quad ... Y_m] \quad m \le d
$$

即为原特征矩阵 $$X$$ 的降维变换，它不仅维度更小，而且具备更强的特征区分度，算是一种对原特征的优化。

然后我们用主成分分析方法跑一个实际的例子，使用著名的 [鸢尾花数据集](https://archive.ics.uci.edu/ml/datasets/iris) 。这个数据集包含 3 种鸢尾花的子属，记录了花瓣与花萼的长宽尺寸作为特征。

![](/resources/2017-12-21-principal-components-analysis/feature.png)
（单独使用一种特征绘制的散点图，不同颜色代表不同种类）

分别计算这四个特征的方差值为

$$
\begin{aligned}
Var(SL) &= 0.685694\\
Var(SW) &= 0.188004\\
Var(PL) &= 3.11318\\
Var(PW) &= 0.582414
\end{aligned}
$$

这里使用 $$SL, SW, PL,PW$$ 分别表示 sepal length, sepal width, petal length, petal width。可以发现，虽然从图上看，petal width 的区分度要明显高于 sepal length，但是它们的方差却是反过来的。细想一下，造成这一现象的原因在于我们没有统一两个特征的度量，这样的话，数值上较大的特征计算的方差肯定会大一点，于是将这些特征归一化处理之后（也就是对每类特征，减去平均值，再除以最大值），再来看看方差

$$
\begin{aligned}
Var(SL) &= 0.162107\\
Var(SW) &= 0.103771\\
Var(PL) &= 0.315483\\
Var(PW) &= 0.343918
\end{aligned}
$$

这就要显得合理许多。

![](/resources/2017-12-21-principal-components-analysis/feature_1.png)
（数据归一化之后的特征分布）

然后将归一化的特征代入主成分分析算法，得到的四阶主成分

![](/resources/2017-12-21-principal-components-analysis/PCA.png)

它们的方差（或者说协方差矩阵的特征值）分别为

$$
\begin{aligned}
Var(Y_1) &= 0.785774\\
Var(Y_2) &= 0.10246\\
Var(Y_3) &= 0.0307906\\
Var(Y_4) &= 0.00600804
\end{aligned}
$$

可以看到其中第一主成分占据了相当大的一部分，其他成分显得更加次要。通过主成分分析，我们找到了训练特征中最具差异化的特征表达，它或许不具备对应的现实意义，例如花瓣的宽度，只是这些特征的加权组合，却能够更好地表达特征。
