---
layout: post
title: SMO 算法介绍
tags: 支持向量机
modify_date: 2018-03-05
---

根据前面对支持向量机的介绍，需要求解的优化问题为

$$
\max\limits_\alpha \quad W(\alpha) = - \frac 1 2 \sum_{i,j = 1}^n \alpha_i \alpha_j y_i y_j<x^{(i)}, x^{(j)}>+ \sum_{i=1}^n \alpha_i
$$

$$
\begin{aligned}
s.t. \quad & \sum_{i = 1}^n \alpha_i y_i = 0\\
&y_i (\omega^T x^{(i)} + b) > 1 \quad \Rightarrow \quad \alpha_i = 0\\
&y_i (\omega^T x^{(i)} + b) = 1 \quad \Rightarrow \quad 0 < \alpha_i < C\\
&y_i (\omega^T x^{(i)} + b) < 1 \quad \Rightarrow \quad \alpha_i = C\\
\end{aligned}
$$

本篇主要阐述 J.C.Platt 在论文中给出的求解上述问题的高效算法，即序列最小化优化(SMO)。对于单变量无约束问题，我们可以沿梯度方向变化自变量来试探求解函数极值。而对于多变量无约束优化，也可以采用类似的方法，但是每次只针对一个变量进行优化，其他变量都固定不变，也就是说此时是一个单变量优化子问题。只要函数在每个维度上均表现出凸性，使用这种方式就一定能得到全局最优值。

### 从 KKT 条件导出的拉格朗日乘子边界

下面要讨论的则属于有约束多变量优化问题，我们仍仿照前面的思路，假设每次只优化一个变量。但这里的困难是，这些变量需要满足约束 \\( \sum_{i = 1}^n \alpha_i y_i = 0\\) ，如果有一个变量变化了，显然这种约束就被打破了。一种巧妙的解决思路是：每次更新两个变量，假设为 \\( \alpha_1, \alpha_2 \\) ，而其他变量保持不变，根据约束条件，两者之间应该满足关系

$$
\alpha_1 y_1 + \alpha_2 y_2 = -\sum_{i=3}^n \alpha_i y_i = \zeta
$$

使用 \\( \alpha_1 \\) 来表示 \\( \alpha_2 \\)，则有

$$
\alpha_2 = y_2\zeta - y_2 \alpha_1 y_1
$$

如果以 \\( \alpha_1 \\) 为横轴， \\( \alpha_2 \\) 为纵轴，那么根据，\\( y_1 y_2 \\) 的符号情况，能分为下图所示的两种函数图象

![](/resources/2017-08-24-smo-algorithm/y1y2.png)

除了上面的关系，\\( \alpha_1, \alpha_2 \\) 还应满足条件 \\( 0 \le \alpha_i \le C \\) 。于是根据具体 \\( y_2 \zeta \\) 的值，对 \\( \alpha_1, \alpha_2 \\) 的约束又可以细分为四种情况：

1、 当 \\( y_1y_2 = 1, \quad y_2\zeta  > C \\) 时

![](/resources/2017-08-24-smo-algorithm/case1.png)

由于两个参数都必须在 0 至 C 之间，并且还满足直线约束条件，那么这两个参数的可能取值就必须在如上图右边所示的加粗线段区域，设其下界为 L ，上界为 H，则有

$$
\begin{aligned}
L &= y_2\zeta - y_2 C y_1\\
H &= C
\end{aligned}
$$

假设在参数更新之前，\\( \alpha_1, \alpha_2 \\) 的值分别为 \\( \alpha_1^{old}, \alpha_2^{old} \\)，易见

$$
\alpha_1^{old} y_1 + \alpha_2^{old} y_2 = \zeta
$$

代入上面的 L 表达式可以计算得到

$$
\begin{aligned}
L &= y_2\zeta - y_2 C y_1\\
&= y_2 (\alpha_1^{old} y_1 + \alpha_2^{old} y_2) -C \\
&= \alpha_1^{old} + \alpha_2^{old} -C
\end{aligned}
$$

2、 当 \\( y_1y_2 = 1, \quad y_2\zeta  < C \\) 时

![](/resources/2017-08-24-smo-algorithm/case2.png)

和前面类似，这时的上下界分别为

$$
\begin{aligned}
L &= 0\\
H &= \alpha_1^{old} + \alpha_2^{old}
\end{aligned}
$$

3、 当 \\( y_1y_2 = -1, \quad y_2\zeta  < 0 \\) 时

![](/resources/2017-08-24-smo-algorithm/case3.png)

$$
\begin{aligned}
L &= 0\\
H &= y_2 \zeta - y_2 C y_1 \\
&= y_2 (\alpha_1^{old} y_1 + \alpha_2^{old} y_2) -C y_1y_2\\
&= \alpha_2^{old} - \alpha_1^{old}+ C
\end{aligned}
$$

4、 当 \\( y_1y_2 = -1, \quad y_2\zeta  > 0 \\) 时

![](/resources/2017-08-24-smo-algorithm/case4.png)

$$
\begin{aligned}
H &= C\\
L &= \alpha_2^{old} - \alpha_1^{old}
\end{aligned}
$$

当直线没有穿过区域 \\( 0 \le \alpha_1 \le C, 0 \le \alpha_2 \le C \\) 时，显然不满足约束条件，不予考虑。综合以上可以得出结论

当 \\( y_1 = y_2 \\) 时

$$
\begin{aligned}
H &= \min(C, \alpha_1^{old} + \alpha_2^{old})\\
L &= \max(\alpha_1^{old} + \alpha_2^{old} -C, 0)
\end{aligned}
$$

当 \\( y_1 = -y_2 \\) 时

$$
\begin{aligned}
H &= \min(\alpha_2^{old} - \alpha_2^{old} + C, C)\\
L &= \max(0, \alpha_2^{old} - \alpha_1^{old})
\end{aligned}
$$

### 对拉格朗日乘子的更新

以上就是在更新 \\( \alpha_1, \alpha_2 \\) 时候的约束条件。接下来将讨论如何更新这两个值，首先，将目标函数 \\( W(\alpha) \\) 写成由 \\( \alpha_1, \alpha_2 \\) 表示的形式

$$
\begin{aligned}
&\sum_{i,j = 1}^n \alpha_i \alpha_j y_i y_j K(x^{(i)}, x^{(j)}) \\
&= \sum_{j=1}^n\alpha_1 \alpha_j y_1 y_j K(x^{(1)}, x^{(j)}) \\
&{}+\sum_{j=1}^n\alpha_2 \alpha_j y_2 y_j K(x^{(2)}, x^{(j)}) \\
&{}+\sum_{i = 3,j = 1}^n \alpha_i \alpha_j y_i y_j K(x^{(i)}, x^{(j)}) \\
&= \alpha_1 \alpha_1 y_1 y_1 K(x^{(1)}, x^{(1)}) + \alpha_1 \alpha_2 y_1 y_2 K(x^{(1)}, x^{(2)}) + \sum_{j=3}^n \alpha_1 \alpha_j y_1 y_j K(x^{(1)}, x^{(j)}) \\
&{}+ \alpha_2 \alpha_1 y_2 y_1 K(x^{(2)}, x^{(1)}) + \alpha_2 \alpha_2 y_2 y_2 K(x^{(2)}, x^{(2)}) + \sum_{j=3}^n\alpha_2 \alpha_j y_2 y_j K(x^{(2)}, x^{(j)})\\
&+ \sum_{i = 3}^n \alpha_i \alpha_1 y_i y_1 K(x^{(i)}, x^{(1)}) \\
&+ \sum_{i = 3}^n \alpha_i \alpha_2 y_i y_2 K(x^{(i)}, x^{(2)}) \\
&+ \sum_{i = 3,j = 3}^n \alpha_i \alpha_j y_i y_j K(x^{(i)}, x^{(j)}) \\
&= k_{11} \alpha_1^2 + k_{22} \alpha_2^2 + 2 y_1y_2 k_{12} + 2\alpha_1 y_1 \sum_{j=3}^n \alpha_j y_j k_{1j} + 2\alpha_2 y_2\sum_{j=3}^n \alpha_j y_j k_{2j} \\
&+ \sum_{i = 3,j = 3}^n \alpha_i \alpha_j y_i y_j k_{ij}\\
&= k_{11}\alpha_1^2 + k_{22} \alpha_2^2 + 2 s k_{12} + 2 \alpha_1 y_1 v_1 +2 \alpha_2 y_2 v_2 + C2
\end{aligned}
$$

由于公式比较繁琐，所以上式用了不少符号替代式子，例如

$$
\begin{aligned}
k_{ij} &= K(x^{(i)}, x^{(j)})\\
s &= y_1y_2\\
v_i &= \sum_{j=3}^n \alpha_j y_j k_{ij}\\
C2 &=\sum_{i = 3,j = 3}^n \alpha_i \alpha_j y_i y_j k_{ij}
\end{aligned}
$$

由于自索引号3 之后的 \\( \alpha \\) 不参与本次更新，所以 C2 应该视为常数。将化简后的式子代入目标函数，可得

$$
W(\alpha_1, \alpha_2) = -\frac 1 2 k_{11} \alpha_1^2 - \frac 1 2 k_{22} \alpha_2^2 - s k_{12} \alpha_1 \alpha_2 -v_1 y_1\alpha_1 - v_2 y_2 \alpha_2 + \alpha_1 + \alpha_2 + C'
$$

然后进一步，使用 \\( \alpha_2 \\) 代替 \\(\alpha_1 \\)

$$
\alpha_1 = y_1 \zeta - s \alpha_2
$$

得到目标函数关于 \\( \alpha_2 \\) 的二次函数

$$
\begin{aligned}
W(\alpha_2) &= - \frac 1 2 (k_{11} + k_{22} - 2 k_{12}) \alpha_2^2 \\
&+ [1 - v_2 y_2 + s(v_1 y_1 - 1) + s y_1 \zeta (k_11 - k_12)] \alpha_2 \\
&- \frac 1 2 y_1 \zeta (k_{11} y_1 \zeta + 2v_1 y_1 - 2)
\end{aligned}
$$

我们的目标是尽量最大化 \\( W(\alpha_2) \\) ，对于二次函数来说，这是再简单不过的事了，先求得其二阶导数

$$
\eta = W''(\alpha_2) = - k_{11} - k_{22} + 2 k_{12}
$$

下面需要分两种情况讨论：

1、如果 \\( \eta \ge 0 \\) ，那么 \\( W(\alpha_2) \\) 是一个凸函数，或者直线，其最大值显然应该在约束边界上取得，也就是说

$$
\alpha_2^{new} = arg \max(W(H), W(L))
$$

2、而如果 \\( \eta < 0 \\)，即 \\( W(\alpha_2) \\) 为凹函数，那么其最大值就可能出现在如下图所示的三个位置

![](/resources/2017-08-24-smo-algorithm/quadri.png)

即位于抛物线对称轴上，或者左边界上，或者右边界上。至于具体是哪种情况很好判断，只需要计算出对称轴的位置，而这只需要对 \\( W(\alpha_2) \\) 求一阶导数，并计算当一阶导数为0 时的 \\( \alpha_2 \\)

$$
\alpha_2^{new} = \frac{\zeta y_2 (k_{11} - k_{12}) + y_2 (v_1 - v_2) - s+ 1} {k_{11} + k_{22} - 2 k_{12}}
$$

然后按照如下条件修正

$$
\alpha_2^{new} = \left \lbrace
\begin{aligned}
&H \quad \alpha_2 \ge H\\
&\alpha_2 \quad L < \alpha_2 < H \\
&L \quad \alpha_2 \le L
\end{aligned}
 \right.
$$

获得 \\(\alpha_2\\) 的更新值后，即可更新 \\(\alpha_1\\)

$$
\alpha_1^{new} = y_1 \zeta - s \alpha_2^{new}
$$

由于 \\(\alpha_2\\) 的更新过程始终满足约束条件，所以 \\(\alpha_1\\) 也必然满足约束。到目前为止，我们叙述了两个参数 \\(\alpha_1, \alpha_2\\) 的更新方法，但比较繁琐的是 \\(\alpha_2\\) 的计算过程，它需要遍历所有样本，因此还需要进一步改进。在这之前我们先来看看最初考虑的超平面

$$
\omega^T x+b=0    
$$

这是对数据进行分类的依据，其判别方法是将特征 \\(x^{(i)}\\) 代入函数

$$
f(x^{(i)}) = \omega^T x^{(i)}+b
$$

从而根据函数值 \\(f(x^{(i)})\\) 的符号对特征进行归类。在训练的过程中，将训练数据代入进去其分类结果显然是不准确的，但由于我们知道训练数据的真实类别，所以可以定义当前参数下的误差

$$
E_i= f(x_i) - y_i
$$

另一方面，根据[上一篇](/2017/07/31/svm-fundamentals.html)的推导，我们知道

$$
\omega = \sum_{i=1}^n \alpha_i y_i x^{(i)}
$$

代入到 \\(f(x)\\) 则有

$$
\begin{aligned}
f(x^{(j)}) &=\sum_{i=1}   ^n \alpha_iy_i <x^{(i)}, x^{(j)} >+b\\
&=\alpha_1 y_1 k_{1j}+\alpha_2 y_2 k_{2j} + \sum_{i=3}   ^n \alpha_iy_i <x^{(i)}, x^{(j)} >+b
\end{aligned}
$$

再根据前面 \\(v_i\\) 的定义，我们可知

$$
v_i=f(x^{(i)}) - \alpha_1 y_1 k_{1i} - \alpha_2 y_2 k_{2i} -b
$$

然后代入到 \\(\alpha_2\\) 的更新公式中，并展开 \\(\zeta\\)

$$
    \begin{aligned}
&\alpha_2^{new}(k_{11} + k_{22} - 2 k_{12}) \\
&= \zeta y_2 (k_{11} - k_{12}) + y_2 (v_1 - v_2) - s+ 1\\
&=(\alpha_1 y_1 + \alpha_2 y_2)y_2 (k_{11} - k_{12}) \\
&+y_2 \left(f(x^{(1)}) - f(x^{(2)} ) - \alpha_1 y_1( k_{11}- k_{12}) - \alpha_2 y_2( k_{21} - k_{22})\right)-s+1
\\
&=\alpha_2 (k_{11}-2k_{12} + k_{22}) + y_2 \left(f(x^{(1)}) - f(x^{(2)})\right) - s+1\\
&=\alpha_2 (k_{11}-2k_{12} + k_{22}) + y_2 \left(f(x^{(1)}) - f(x^{(2)}) + y_2 - y_1\right)
\end{aligned}
$$

也就是说

$$
\alpha_2^{new} = \alpha_2 - \frac { y_2 \left(f(x^{(1)}) - f(x^{(2)}) + y_2 - y_1\right) }{\eta}   
$$

而根据误差 \\(E_i\\) 定义，又可将上式改写为

$$
\alpha_2^{new} = \alpha_2 - y_2\frac {E_1 - E_2 }{\eta}   
$$

于是，对 \\(\alpha_2\\) 的更新转换成了增量形式，在程序中，我们只需要将误差项存储起来，就能立即获得更新值。对于 \\(\alpha_1\\) 来说，仍然展开 \\(\zeta\\)

$$
    \begin{aligned}
\alpha_1^{new} &= y_1 (\alpha_1 y_1 + \alpha_2 y_2) - s \alpha_2^{new}\\
&=\alpha_1 - s(\alpha_2^{new} - \alpha_2)
\end{aligned}
$$

### 对误差项的更新

既然 \\(\alpha_2\\) 的更新依赖于误差项 \\(E_1, E_2\\) ，那么我们也应该对它们进行更新，根据定义

$$
    \begin{aligned}
E_i &= f(x^{(i)}) -y_i\\
&=\alpha_1 y_1 k_{i1} + \alpha_2 y_2 k_{i2} + \sum_{j=3}^n \alpha_j y_j k_{ij}+b -y_i
\end{aligned}
    $$

这是更新前的值，而在更新之后

$$
\begin{aligned}    
E_i^{new} &= \alpha_1^{new} y_1 k_{i1} + \alpha_2^{new} y_2 k_{i2} + \sum_{j=3}^n \alpha_j y_j k_{ij}+b^{new} -y_i\\
&=\alpha_1^{new} y_1 k_{i1} + \alpha_2^{new} y_2 k_{i2} -\alpha_1 y_1 k_{i1} - \alpha_2 y_2 k_{i2} + b^{new}-b+\sum_{j=1}^n \alpha_j y_j k_{ij}+b -y_i\\
&=E_i +\alpha_1^{new} y_1 k_{i1} + \alpha_2^{new} y_2 k_{i2} -\alpha_1 y_1 k_{i1} - \alpha_2 y_2 k_{i2}+b^{new}-b\\
&= E_i + (\alpha_1^{new} - \alpha_1)y_1 k_{i1} + (\alpha_2^{new} - \alpha_2)y_2 k_{i2}  + b^{new} - b
\end{aligned}
$$

### 对 b 的更新

注意这里我们还引入了对 \\(b\\) 的更新，我们看到在 \\(W(\alpha)\\) 中，并没有 b 这一项，便也没有什么存在感，但是它又确实影响了分类器的位置，所以 b 的更新应该单独计算。现在我们再回到 KKT 条件

$$
\begin{aligned}
&y_i (\omega^T x^{(i)} + b) > 1 \quad \Rightarrow \quad \alpha_i = 0\\
&y_i (\omega^T x^{(i)} + b) = 1 \quad \Rightarrow \quad 0 < \alpha_i < C\\
&y_i (\omega^T x^{(i)} + b) < 1 \quad \Rightarrow \quad \alpha_i = C\\
\end{aligned}
$$

可以发现，b的值与 \\(\alpha\\) 通过 KKT 条件建立起了联系，如果 \\(0 < \alpha_1 < C\\)，那就有

$$
    y_1 (\omega^T x^{(1)}+ b) = 1
$$

将 b 解出来后

$$
    \begin{aligned}
b &= y_1 -\omega^T x^{(1)}    \\
&= y_1 - \sum_{i=3} \alpha_i y_i k_{i1}-\alpha_1 y_1 x^{(1)}-\alpha_2 y_2 x^{(2)}
\end{aligned}
$$

而 \\(\alpha_1, \alpha_2\\) 更新之后，如果仍然有 \\(0 < \alpha_1^{new} < C\\) ，b 的更新值也必须满足

$$
b^{new}= y_1 - \sum_{i=3} \alpha_i y_i k_{i1}-\alpha_1^{new} y_1 k_{11}-\alpha_2^{new} y_2 k_{21}
$$

考虑到我们在更新 \\(E_i\\) 的时候，获得了如下公式

$$
E_i = \alpha_1 y_1 k_{i1} + \alpha_2 y_2 k_{i2} + \sum_{j=3}^n \alpha_j y_j k_{ij}+b -y_i
$$

于是

$$
 y_1 - \sum_{i=3} \alpha_i y_i k_{i1}= -E_1 +\alpha_1 y_1 k_{11} + \alpha_2 y_2 k_{12}  +b
$$

代入到 \\(b^{new}\\) 即得

$$
    \begin{aligned}
b^{new}&=-E_1 +\alpha_1 y_1 k_{11} + \alpha_2 y_2 k_{12}  +b-\alpha_1^{new} y_1 k_{11}-\alpha_2^{new} y_2 k_{21}\\
&=b -E_1 +(\alpha_1- \alpha_1^{new}) y_1 k_{11}+(\alpha_2-\alpha_2^{new}) y_2 k_{21}
\end{aligned}
$$

另一方面，当 \\(0<\alpha_2^{new}< C\\) 时，同样能得到一个 b 得更新值，为了区分，我们将前一个 \\(b^{new}\\) 记作 \\(b_1\\) ，后一个记作 \\(b_2\\)，即

$$
    \begin{aligned}
b_1=b -E_1 +(\alpha_1- \alpha_1^{new}) y_1 k_{11}+(\alpha_2-\alpha_2^{new}) y_2 k_{21}\\
b_2=b -E_2 +(\alpha_1- \alpha_1^{new}) y_1 k_{12}+(\alpha_2-\alpha_2^{new}) y_2 k_{22}
\end{aligned}
$$

可以看到，这里出现了两种形式的 b，那么 b 的更新值究竟该怎样取呢? 下面我想借助图来作说明

![](/resources/2017-08-24-smo-algorithm/svm.png)

上图是一个简单的支持向量机线性分类问题，根据我们之前的介绍，支持向量机最终将训练出来一条直线（或者说超平面），即图中的蓝色实线，使得所有样本到该直线距离的最小值最大化。图中的蓝色虚线，便是离分类面最近样本点所在的直线，于是虚线与直线的距离就为 \\(\gamma\\) 。

在 KKT 条件中，如果 \\(0<\alpha_1<C\\) ，则意味着样本点 \\(x_1\\) 位于蓝色虚线上，也就是说到分类面的距离等于 \\(\gamma\\)，而如果 \\(0<\alpha_2 < C\\) 也成立，那么同样地， \\(x_2\\) 也位于蓝色虚线上。当我们把这两个点单独拿出来看的时候，就像下图这样

![](/resources/2017-08-24-smo-algorithm/margin.png)

由于 b 确定了分类面的位置，所以这里我们看到，针对不同的样本，我们使用了不同的 b 来表示分类面。但事实上，在同一个训练模型中，上面左右两个图是可以合并在一起的，它们共享同一条蓝色虚线，以及同一个 \\(\gamma\\)，也就意味着分类面也是相同的，所以我们得出重要结论

$$
b_1 = b_2
$$

以上叙述是针对 \\(\alpha_1, \alpha_2\\) 都不在边界上的情况讨论的。而当其中一个参数在边界上时，则该样本点就位于蓝色虚线后面离分类面较远的位置。如下图所示

![](/resources/2017-08-24-smo-algorithm/one-non-bound.png)

显然这种情况下，只需要 \\(b^{new} = b_1\\) 即可。

### 关于目标函数值 W 的更新

根据一开始我们提到的目标函数形式，如果按照公式去计算，由于存在双重循环，其时间复杂度将是 \\(O(n^2)\\)。为了改善性能，我们可以在程序中维护一个变量来存储 W 在每次更新了 \\(\alpha_1, \alpha_2\\) 后的值。在更新之前，W 关于 \\(\alpha_1, \alpha_2\\) 的表达式为

$$
W(\alpha_1, \alpha_2) = -\frac 1 2 k_{11} \alpha_1^2 - \frac 1 2 k_{22} \alpha_2^2 - s k_{12} \alpha_1 \alpha_2 -v_1 y_1\alpha_1 - v_2 y_2 \alpha_2 + \alpha_1 + \alpha_2 + C'
$$

而更新之后

$$
\begin{aligned}
W(\alpha_1^{new}, \alpha_2^{new}) &= -\frac 1 2 k_{11} (\alpha_1^{new})^2 - \frac 1 2 k_{22} (\alpha_2^{new})^2 - s k_{12} \alpha_1^{new} \alpha_2^{new} \\
&-v_1 y_1\alpha_1^{new} - v_2 y_2 \alpha_2^{new} + \alpha_1^{new} + \alpha_2^{new} + C'\\
&=-\frac 1 2 k_{11} \left((\alpha_1^{new})^2-\alpha_1^2)\right) - \frac 1 2 k_{22} \left((\alpha_2^{new})^2 -\alpha_2^2\right)\\
&- s k_{12} (\alpha_1^{new} \alpha_2^{new}-\alpha_1 \alpha_2) -v_1 y_1(\alpha_1^{new} -\alpha_1)- v_2 y_2( \alpha_2^{new} -\alpha_2)\\
&+ \alpha_1^{new}-\alpha_1  + \alpha_2^{new}-\alpha_2 + W(\alpha_1, \alpha_2)
\end{aligned}
$$

其中对于 \\(v_i\\) 来讲

$$
    \begin{aligned}
v_i &= f(x^{(i)}) - b - \alpha_1 y_1 k_{11} - \alpha_2 y_2 k_{22}\\
&= E_i + y_i - b - \alpha_1 y_1 k_{11} - \alpha_2 y_2 k_{22}
\end{aligned}
$$

这样我们便将 W 写成了增量形式，在每次迭代的时候进行更新，具有常数级别的时间复杂度。
