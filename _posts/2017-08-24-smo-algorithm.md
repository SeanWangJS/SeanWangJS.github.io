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

除了上面的关系，$$\alpha_1, \alpha_2$$ 还应满足条件 $$0 \le \alpha_i \le C$$。于是根据具体 $$y^{(2)} \zeta$$ 和 $$y^{(1)}y^{(2)}$$ 的值，对 $$\alpha_1, \alpha_2$$ 的约束可以分为下图所示的四种情况：

![](/resources/2017-08-24-smo-algorithm/constraint.jpg)

其中的 $$H, L$$ 分别是上界和下界。假设 $$\alpha_1, \alpha_2$$ 在更新之前的值







