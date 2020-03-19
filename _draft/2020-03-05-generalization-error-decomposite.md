---
title: 泛化误差分解
tags: 统计学习 泛化误差 机器学习
---

机器学习模型的泛化性能是指其应用于训练集之外数据的预测能力，如果模型在训练集上有很好的表现，但泛化性能却很糟糕，便可以认为模型已经过拟合了。为了量化模型的泛化性能，我们定义**泛化误差**为**模型在总体数据上的期望损失**，显然泛化误差越小，代表模型有更好的泛化能力。

假设总体数据空间为 \(\Omega\)，对于其中的一个样本集 \(D \subset \Omega\)，训练得到模型 \(f(x; D)\)，下面简写成 \(f(x)\)，再定义损失函数 \(L(y, f(x))\)，那么泛化误差可以表示为 
\[
  Err = \iint_{\Omega} p(x, y)L(y, f(x)) \mathrm{d}x\mathrm{d}y
  \]

如果限定到回归问题，并采用平方损失函数，则

\[
  Err = \iint_{\Omega} p(x, y) \left(y - f(x)\right)^2\mathrm{d}x\mathrm{d}y
  \]

上式可以看作 \(Err\) 对 \(f(x)\) 的泛函，假定当 \(f(x) = f^*(x)\) 时，\(Err\) 可以取到最小值，并将 \(f(x)\) 拆分成两个部分 \(f(x) = f^*(x) + \epsilon\eta(x) \)，这里的 \(\eta(x)\) 可以是任意函数，\(\epsilon\) 是它的系数。于是
\[
    Err(\epsilon) = \iint_{\Omega} p(x, y) \left(y - f^*(x) - \epsilon \eta(x)\right)^2\mathrm{d}x\mathrm{d}y
  \]

根据前面的假设， \(\epsilon=0\) 使 \(Err\) 达到极值，也就是说 \(Err\) 对 \(\epsilon\) 的导数在 \(\epsilon = 0\) 时等于 0
\[\frac{\partial Err}{\partial \epsilon} \mid_{\epsilon=0} = 0\]

具体计算一下

\[
  \begin{aligned}&-2\iint_{\Omega} p(x, y)(y - f^*(x) ) \eta(x) \mathrm{d}x\mathrm{d}y = 0 \\
  &\Rightarrow \int \left(\int p(x, y)(y-f^*(x)) \mathrm{d}y\right) \eta(x)\mathrm{d}x = 0
  \end{aligned}
  \]

由于对于任意的 \(\eta(x)\)，上式都成立，所以，内部积分应该始终为零
\[
  \begin{aligned}
  &\int p(x, y)(y-f^*(x)) \mathrm{d}y = 0 \\
  &\Rightarrow \int p(x, y) y \mathrm{d}y = f^*(x)\int p(x, y) \mathrm{d}y\\
  &\Rightarrow f^*(x) = \frac{\int p(x, y) y \mathrm{d}y}{p(x)} = \int p(y\mid x) y \mathrm{d}y
  \end{aligned}
  \]

这样我们就得到了使得泛化误差最小的解，也就是最优解，它可以看作是 \(y\) 关于 \(x\) 的条件均值 \(E(y\mid x)\)，这正是回归问题解的定义，也称为回归函数。把泛化误差按下面的方式展开

\[
  \begin{aligned}
  Err&=\iint_{\Omega}p(x,y)(y - f(x))^2  \mathrm{d}x\mathrm{d}y \\
  &= \iint_{\Omega}p(x, y) (y - f^*(x) + f^*(x) - f(x))^2\mathrm{d}x\mathrm{d}y\\
  &= \iint_{\Omega} p(x, y) (y - f^*(x))^2\mathrm{d}x\mathrm{d}y 
  +\iint_{\Omega} p(x, y) (f^*(x) - f(x))^2\mathrm{d}x\mathrm{d}y + \iint_{\Omega} p(x, y)2 (y-f^*(x))(f^*(x)-f(x)) \mathrm{d}x\mathrm{d}y
  \end{aligned}
  \]

其中第二项

\[
  \begin{aligned}\iint_{\Omega} p(x, y) (f^*(x) - f(x))^2\mathrm{d}x\mathrm{d}y &= \int (f^*(x) - f(x))^2 \int p(x, y) \mathrm{d}y \mathrm{d}x\\
  &=\int (f^*(x) - f(x))^2 p(x)\mathrm{d}x 
  \end{aligned}
  \]

第三项

\[
  \begin{aligned}
  &\iint_{\Omega}2 p(x, y) (y-f^*(x))(f^*(x)-f(x)) \mathrm{d}x\mathrm{d}y \\
  &=\iint_{\Omega} 2p(x,y) y f^*(x)\mathrm{d}x\mathrm{d}y -
   \iint_{\Omega} 2p(x, y)y f(x)\mathrm{d}x\mathrm{d}y -
   \iint_{\Omega}2p(x, y) (f^*(x))^2\mathrm{d}x\mathrm{d}y + 
   \iint_{\Omega}2p(x, y) f^*(x)f(x) \mathrm{d}x\mathrm{d}y\\
   &= 2\int \left(\int p(x, y)y\mathrm{d}y\right) f^*(x)\mathrm{d}x - 2 \int \left(\int p(x, y)y\mathrm{d}y\right)f(x)\mathrm{d}x -2 \int \left(\int p(x, y)\mathrm{d}y\right) (f^*(x))^2\mathrm{d}x + 2 \int \left(\int p(x, y)\mathrm{d}y\right) f^*(x) f(x)\mathrm{d}x\\
   &=2 \int f^*(x) p(x) f^*(x)\mathrm{d}x - 2 \int f^*(x) p(x) f(x)  \mathrm{d}x-2\int p(x)(f^*(x))^2\mathrm{d}x + 2 \int p(x) f^*(x)f(x)
  \mathrm{d}x\\
  &=0
  \end{aligned}
  \]

于是 
\[
  Err = \int (f^*(x) - f(x))^2 p(x)\mathrm{d}x + \iint_{\Omega} p(x, y) (y - f^*(x))^2\mathrm{d}x\mathrm{d}y 
  \]

对于一个特定的训练集 \(D\) 得到的模型 \(f(x;D)\)，它的泛化误差就是

\[
  Err = \int (f^*(x) - f(x;D))^2 p(x)\mathrm{d}x + \iint_{\Omega} p(x, y) (y - f^*(x))^2\mathrm{d}x\mathrm{d}y 
  \]

可以发现上式第二项和具体的模型无关，现令其为 \(\xi\)

\[
  \xi = \iint_{\Omega} p(x, y) (y - f^*(x))^2\mathrm{d}x\mathrm{d}y 
  \]

也就是说无论采用什么方法优化，\(Err\) 的值都不可能小于第二项的值。假设，对于所有的 \(x\) 都有 \(y = f^*(x)\) ，也就是说，\(\Omega\) 中的所有点都刚好落到回归函数上，这时有 \(\xi = 0\)，那么什么情况下这种假设成立呢？我们知道 

\[
  f^*(x) = E[y\mid x]
  \]

也就是说，对于任意 \(a\)，在 \(\Omega\) 上找到所有 \(x = a\) 的点 

\[
  (a, y_{a, 1}), (a, y_{a,2}), (a, y_{a,3}) ....
  \]

\(f^*(x)\) 等于这些点 \(y_{a,1}, y_{a,2},..\) 的平均值。而要使假设成立，则必然有 

\[
  y_{a,1} = y_{a,2} = y_{a,3} = ... = f^*(x)
  \]

但在现实情况中，这样的等式几乎是不可能成立的，总会存在一些样本，它们的 \(x\) 相同，但却有不同的 \(y\)，这种情况我们一般称作**噪声**。所以，正是由于噪声的存在使得 \(\xi\) 不能达到 0，换句话说，\(\xi\) 反映了数据内在的噪声程度。接下来，我们再讨论第一项

\[
   \int (f^*(x) - f(x;D))^2 p(x)\mathrm{d}x
  \]

首先假设一个由有限个训练集合构成的集合 \(\mathbb{S}\)，对于其中的每一个训练集 \(C\subset \mathbb{S}\)，可以得到模型 \(f(x;C)\)，于是对于\(\mathbb{S}\) 来说，得到的平均模型为 \(E_{C\subset\mathbb{S}}[f(x;C)]\)，简写成 \(E[f(x;C)]\)
\[
  E[f(x;C)] = \frac 1 {\mid \mathbb{S}\mid}\sum_{C \in \mathbb{S}} f(x;C)
  \]

用下面的方式展开第一项公式 
\[
  \begin{aligned}
  &\int (f^*(x) - f(x;D))^2 p(x)\mathrm{d}x \\&= \int (f^*(x) -E[f(x;C)] + E[f(x;C)]- f(x;D))^2 p(x)\mathrm{d}x \\
  &= \int (f^*(x) - E[f(x;C)])^2  p(x)\mathrm{d}x + \int (E[f(x;C)] -f(x;D))^2  p(x)\mathrm{d}x + \int (f^*(x) - E[f(x;C)]) (E[f(x;C)] -f(x;D)) p(x)\mathrm{d}x
  \end{aligned}
  \]

其中第三项
\[
   \begin{aligned}
  &\int (f^*(x) - E[f(x;C)]) (E[f(x;C)] -f(x;D)) p(x)\mathrm{d}x\\
  &=
  \end{aligned}
  \]