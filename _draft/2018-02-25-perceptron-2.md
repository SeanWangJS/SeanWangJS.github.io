---
layout: post
title: 感知器算法 II
tags: 神经网络
---

### 线性多分类问题

前面我们讨论了二分类的感知器算法，简单来说，其思想是首先定义一个初始超平面，然后筛选出被错误分类的点，并且对这些点到超平面的距离求和作为目标函数，最后再最小化该函数，使得尽量少的点被错误分类。

而对于多分类问题，可以用类似的思想来解决。首先我们来考虑数据的类别表示，在二分类问题中，使用 1 和 -1 来确定数据所属的类别，而在多分类问题时，则可以使用用 1 和 -1 组成的向量 $$\mathbf{y}$$ 来确定，比如 $$x$$ 属于类别 $$\mathcal{C}_i$$ ，那么就可以将 $$\mathbf{y}$$ 的第 i 个值设为 1，其余值设为 -1，即

$$
\mathbf{y} = \left[
\begin{aligned}
-1\\-1\\\vdots\\1\\\vdots\\-1
\end{aligned}
\right]
$$

如果数据能够被线性分割，那么可以先定义一系列超平面

$$
W^T x+ \mathbf{b} = \mathbf{0}
$$

假设特征维度等于 *d* ，类别数量等于 *m* ，那么 $$W^T$$ 为 m x d 的矩阵， **b** 为 m 维向量。写成矩阵形式就为

$$
\left[
\begin{aligned}
W_1^T\\W_2^T\\\vdots\\W_m^T
\end{aligned}
\right]x
+
\left[
\begin{aligned}
b_1\\b_2\\\vdots\\b_m
\end{aligned}
\right]=\mathbf{0}
$$

也就是说，这是 m 个超平面方程写成的统一形式，这里仍然规定所有的 $$W_i, i = 1,2,,,m$$ 的2范数都为 1。当然，如果将 **b** 并入 *W* 写成增广矩阵形式

$$
\mathbf{W}^T = [\mathbf{b}\quad W^T]\\
\mathbf{x} =\left[
\begin{aligned}
1\\x
\end{aligned}
\right]
$$

那么这些超平面的方程又可写作

$$
\mathbf{W}^T \mathbf{x} = \mathbf{0}
$$

















end

end

end
