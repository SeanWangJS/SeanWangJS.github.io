---
layout: default
---

### SMO 算法

根据前面对支持向量机的介绍，需要求解的优化问题为

$$
\max\limits_\alpha \quad W(\alpha) = - \frac 1 2 \sum_{i,j = 1}^m \alpha_i \alpha_j y^{(i)} y^{(j)}<x^{(i)}, x^{(j)}>+ \sum_{i=1}^m \alpha_i\\
$$

$$
\begin{aligned}
s.t. \quad & \sum_{i = 1}^m \alpha_i y^{(i)} = 0\\
&y^{(i)} (\omega^T x^{(i)} + b) > 1 \quad => \quad \alpha_i = 0\\
&y^{(i)} (\omega^T x^{(i)} + b) = 1 \quad => \quad 0 < \alpha_i < C\\
&y^{(i)} (\omega^T x^{(i)} + b) < 1 \quad => \quad \alpha_i = C\\
\end{aligned}
$$

本篇主要阐述 J.C.Platt 在论文中给出的求解上述问题的高效算法，即序列最小化优化(SMO)。对于单变量无约束问题，我们可以沿梯度方向变化自变量来试探求解函数极值。而对于多变量无约束优化，也可以采用类似的方法，但是每次只针对一个变量进行优化，其他变量都固定不变，也就是说此时是一个单变量优化子问题。只要函数在每个维度上均表现出凸性，使用这种方式就一定能得到全局最优值。

而下面要讨论的则属于有约束多变量优化问题，我们仍仿照前面的思路，每次只优化一个变量。但这里的困难是，这许多变量需要满足约束 $$\sum_{i = 1}^m \alpha_i y^{(i)} = 0$$，如果有一个变量变化了，显然这种约束就很可能被打破。一种巧妙的解决思路是：每次更新两个变量，假设为 $$\alpha_1, \alpha_2$$ ，而其他变量保持不变，根据约束条件，两者之间应该满足关系 

$$
\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = -\sum_{i=3}^m \alpha_i y^{(i)} = \zeta
$$

使用 $$\alpha_1$$ 来表示 $$\alpha_2$$，则有 

$$
\alpha_2 = y^{(2)}\zeta - y^{(2)} \alpha_1 y^{(1)}
$$

如果以 $$\alpha_1$$ 为横轴， $$\alpha_2$$ 为纵轴，那么根据，$$y^{(1)} y^{(2)}$$ 的情况，能分为下图所示的两种函数图象

![](/resources/2017-08-24-smo-algorithm/y1y2.png)

除了上面的关系，$$\alpha_1, \alpha_2$$ 还应满足条件 $$0 \le \alpha_i \le C$$。于是根据具体 $$y^{(2)} \zeta$$ 的值，对 $$\alpha_1, \alpha_2$$ 的约束又可以细分为四种情况：

1、 当 $$y^{(1)}y^{(2)} = 1, \quad y^{(2)}\zeta  > C$$ 时

![](/resources/2017-08-24-smo-algorithm/case1.png)

由于两个参数都必须在 0 至 C 之间，并且还满足直线约束条件，那么这两个参数的可能取值就必须在如上图右边所示的加粗线段区域，设其下界为 L ，上界为 H，则有

$$
\begin{aligned}
L &= y^{(2)}\zeta - y^{(2)} C y^{(1)}\\
H &= C
\end{aligned}
$$

假设在参数更新之前，$$\alpha_1, \alpha_2$$ 的值分别为 $$\alpha_1^{old}, \alpha_2^{old}$$，易见

$$
\alpha_1^{old} y^{(1)} + \alpha_2^{old} y^{(2)} = \zeta
$$

代入上面的 L 表达式可以计算得到 

$$
\begin{aligned}
L &= y^{(2)}\zeta - y^{(2)} C y^{(1)}\\
&= y^{(2)} (\alpha_1^{old} y^{(1)} + \alpha_2^{old} y^{(2)}) -C \\
&= \alpha_1^{old} + \alpha_2^{old} -C
\end{aligned}
$$

2、 当 $$y^{(1)}y^{(2)} = 1, \quad y^{(2)}\zeta  < C$$ 时

![](/resources/2017-08-24-smo-algorithm/case2.png)

和前面类似，这时的上下界分别为

$$
\begin{aligned}
L &= 0\\
H &= \alpha_1^{old} + \alpha_2^{old}
\end{aligned}
$$

3、 当 $$y^{(1)}y^{(2)} = -1, \quad y^{(2)}\zeta  < 0$$ 时

![](/resources/2017-08-24-smo-algorithm/case3.png)

$$
\begin{aligned}
L &= 0\\
H &= y^{(2)} \zeta - y^{(2)} C y^{(1)} \\
&= y^{(2)} (\alpha_1^{old} y^{(1)} + \alpha_2^{old} y^{(2)}) -C y^{(1)}y^{(2)}\\
&= \alpha_2^{old} - \alpha_1^{old}+ C
\end{aligned}
$$

4、 当 $$y^{(1)}y^{(2)} = -1, \quad y^{(2)}\zeta  > 0$$ 时

![](/resources/2017-08-24-smo-algorithm/case4.png)

$$
\begin{aligned}
H &= C\\
L &= \alpha_2^{old} - \alpha_1^{old}
\end{aligned}
$$

当直线没有穿过区域 $$0 \le \alpha_1 \le C, 0 \le \alpha_2 \le C$$ 时，显然不满足约束条件，不予考虑。综合以上可以得出结论

当 $$y^{(1)} = y^{(2)}$$ 时

$$
\begin{aligned}
H &= \min(C, \alpha_1^{old} + \alpha_2^{old})\\
L &= \max(\alpha_1^{old} + \alpha_2^{old} -C, 0)
\end{aligned}
$$

当 $$y^{(1)} = -y^{(2)}$$ 时

$$
\begin{aligned}
H &= \min(\alpha_2^{old} - \alpha_2^{old} + C, C)\\
L &= \max(0, \alpha_2^{old} - \alpha_1^{old} -C)
\end{aligned}
$$

以上就是在更新 $$\alpha_1, \alpha_2$$ 时候的约束条件。接下来将讨论如何更新这两个值，首先，将目标函数 $$W(\alpha)$$ 写成由 $$\alpha_1, \alpha_2$$ 表示的形式

$$
\begin{aligned}
&\sum_{i,j = 1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} K(x^{(i)}, x^{(j)}) \\
&= \sum_{j=1}^m\alpha_1 \alpha_j y^{(1)} y^{(j)} K(x^{(1)}, x^{(j)}) \\
&{}+\sum_{j=1}^m\alpha_2 \alpha_j y^{(2)} y^{(j)} K(x^{(2)}, x^{(j)}) \\
&{}+\sum_{i = 3,j = 1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} K(x^{(i)}, x^{(j)}) \\
&= \alpha_1 \alpha_1 y^{(1)} y^{(1)} K(x^{(1)}, x^{(1)}) + \alpha_1 \alpha_2 y^{(1)} y^{(2)} K(x^{(1)}, x^{(2)}) + \sum_{j=3}^m \alpha_1 \alpha_j y^{(1)} y^{(j)} K(x^{(1)}, x^{(j)}) \\
&{}+ \alpha_2 \alpha_1 y^{(2)} y^{(1)} K(x^{(2)}, x^{(1)}) + \alpha_2 \alpha_2 y^{(2)} y^{(2)} K(x^{(2)}, x^{(2)}) + \sum_{j=3}^m\alpha_2 \alpha_j y^{(2)} y^{(j)} K(x^{(2)}, x^{(j)})\\
&+ \sum_{i = 3}^m \alpha_i \alpha_1 y^{(i)} y^{(1)} K(x^{(i)}, x^{(1)}) \\
&+ \sum_{i = 3}^m \alpha_i \alpha_2 y^{(i)} y^{(2)} K(x^{(i)}, x^{(2)}) \\
&+ \sum_{i = 3,j = 3}^m \alpha_i \alpha_j y^{(i)} y^{(j)} K(x^{(i)}, x^{(j)}) \\
&= k_{11} \alpha_1^2 + k_{22} \alpha_2^2 + 2 y^{(1)}y^{(2)} k_{12} + 2\alpha_1 y^{(1)} \sum_{j=3}^m \alpha_j y^{(j)} k_{1j} + 2\alpha_2 y^{(2)}\sum_{j=3}^m \alpha_j y^{(j)} k_{2j} \\
&+ \sum_{i = 3,j = 3}^m \alpha_i \alpha_j y^{(i)} y^{(j)} k_{ij}\\
&= k_{11}\alpha_1^2 + k_{22} \alpha_2^2 + 2 s k_{12} + 2 \alpha_1 y^{(1)} v_1 +2 \alpha_2 y^{(2)} v_2 + C
\end{aligned}
$$

由于公式比较繁琐，所以上式用了不少符号替代式子，例如 

$$
\begin{aligned}
k_{ij} &= K(x^{(i)}, x^{(j)})\\
s &= y^{(1)}y^{(2)}\\
v_i &= \sum_{j=3}^m \alpha_j y^{(j)} k_{ij}\\
C &=\sum_{i = 3,j = 3}^m \alpha_i \alpha_j y^{(i)} y^{(j)} k_{ij}

\end{aligned}
$$

由于自索引号3 之后的 $$\alpha$$ 不参与本次更新，所以 C 应该视为常数。

将化简后的式子代入目标函数，可得

$$
W(\alpha_1, \alpha_2) = -\frac 1 2 k_{11} \alpha_1^2 - \frac 1 2 k_{22} \alpha_2^2 - s k_{12} \alpha_1 \alpha_2 -v_1 y^{(1)}\alpha_1 - v_2 y^{(2)} \alpha_2 + \alpha_1 + \alpha_2 + C'
$$

然后进一步，使用 $$\alpha_2$$ 代替 $$\alpha_1$$

$$
\alpha_1 = y^{(1)} \zeta - s \alpha_2
$$

得到目标函数关于 $$\alpha_2$$ 的二次函数

$$
\begin{aligned}
W(\alpha_2) &= - \frac 1 2 (k_{11} + k_{22} - 2 k_{12}) \alpha_2^2 \\
&+ [1 - v_2 y^{(2)} + s(v_1 y^{(1)} - 1) + s y^{(1)} \zeta (k_11 - k_12)] \alpha_2 \\
&- \frac 1 2 y^{(1)} \zeta (k_{11} y^{(1)} \zeta + 2v_1 y^{(1)} - 2)
\end{aligned}
$$

我们的目标是尽量最大化 $$W(\alpha_2)$$ ，对于二次函数来说，这是再简单不过的事了，先求得其二阶导数

$$
\eta = W''(\alpha_2) = - k_{11} - k_{22} + 2 k_{12}
$$

下面需要分两种情况讨论：

1、如果 $$\eta >= 0$$ ，那么 $$W(\alpha_2)$$ 是一个凹函数，或者直线，其最大值显然应该在约束边界上取得，也就是说

$$
\alpha_2^{new} = arg \max(W(H), W(L))
$$

2、而如果 $$\eta < 0$$，即 $$W(\alpha_2)$$ 为凸函数，那么其最大值就可能出现在如下图所示的三个位置

![](/resources/2017-08-24-smo-algorithm/quadri.png)

即位于对称线上，或者左边界上，或者右边界上。至于具体是哪种情况很好判断，只需要计算出对称轴的位置，而这只需要对 $$W(\alpha_2)$$ 求一阶导数，并计算当一阶导数为0 时的 $$\alpha_2$$

$$
\alpha_2 = \frac{\zeta y^{(2)} (k_{11} - k_{12}) + y^{(2)} (v_1 - v_2) - s+ 1} {k_{11} + k_{22} - 2 k_{12}}
$$

然后按照如下条件更新

$$
\alpha_2^{new} = \left \lbrace 
\begin{array}{c}
H \quad \alpha_2 \ge H\\
\alpha_2 \quad L < \alpha_2 < H \\
L \quad \alpha_2 \le L
\end{array}
 \right.
$$

