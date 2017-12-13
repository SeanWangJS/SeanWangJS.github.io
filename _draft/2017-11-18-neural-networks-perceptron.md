---
layout: default
---

## 神经网络：单层感知器

感知器这个名称听起来就有点玄，再加一些教材并不想好好说话，就更加令人费解这一个概念。其实所谓的感知器就是一系列分类面，而单层感知器则不过是二维面上的一条直线，或者高维空间的超平面，它具有统一的形式

$$
\mathbf{w}^T \mathbf{x} + b = 0
$$

其中 $\mathbf{w} = [w_1 \quad w_2 \quad ... w_m],  \quad\mathbf{x} = [x_1 \quad x_2 \quad ... x_m]$，由于 $\mathbf{w}$ 定义了超平面的法线方向，对其进行任意缩放不会改变方向，所以规定 $\|\mathbf{w}\| = 1$ 也是合理的，并有助于后续分析。


定义函数

$$
r(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b
$$

这个函数的作用是，将特征向量 $\mathbf{x}$ 映射到一个值上，而这个值的大小正是点 $\mathbf{x}$ 到超平面 $\mathbf{w}^T \mathbf{x} + b = 0$ 的距离，其正负号则表明点位于超平面的哪一侧。

假设有一个超平面能完美地将两类样本分割，设其方程为

$$
\mathbf{w}^{* T} \mathbf{x} +b^* =0
$$

那么将所有特征点 $\{\mathbf x_i | i = 1,2,3,,,n\}$ 依次代入函数

$$
r(\mathbf{x}) = \mathbf{w}^{* T} \mathbf{x} +b^*
$$

将会得到一组距离 $\bar r_1, \bar r_2,\bar r_3,,,\bar r_n$。将这组值作为标定量，那么我们寻找分隔面的任务其实就是寻找 $\mathbf{w}, b$ ，使得将特征点代入距离函数之后能够得到标定距离的过程。而对于任意的参数 $\mathbf{w}, b$ ，我们可以定义总体误差函数

$$
e(\mathbf{w},b) = \frac 1 2 \sum_{i=1}^n (\mathbf{w}^T \mathbf{x}^{(i)} + b -\bar r_i)^2
$$

可以看到，如果 $\mathbf{w}, b$ 是趋近完美的（也就是能分开样本特征），那么上述误差也是趋近于 0 的。于是我们的问题就变成了求解最优化问题

$$
\min_{\mathbf{w},b} \quad e(\mathbf{w},b)
$$

这是一个无约束优化问题，使用梯度下降法的迭代格式为

$$
\mathbf w^{new} = \mathbf{w}^{old} - \eta \nabla_{\mathbf{w}} e
$$

其中

$$
\nabla_{\mathbf w}e =\frac 1 2 \sum_{i=1}^n 2(\mathbf{w}^T \mathbf x^{(i)} + b -\bar r_i) \mathbf{x}^{(i)} = \sum_{i=1}^n (r_i -\bar r_i)\mathbf x^{(i)}
  $$

于是就得到

$$
\mathbf w^{new} = \mathbf{w}^{old} - \eta\sum_{i=1}^n (r_i -\bar r_i)\mathbf x^{(i)}
$$

对于相当大的训练集，如果每次迭代都要计算所有样本，就会显得十分笨重，可以考虑把样本集进行分割，每次迭代仅使用少量样本，即可显著减少计算量。更极端的情况是，每次仅计算一个样本，便可得到如下的轻量级迭代格式

$$
\mathbf w^{new} = \mathbf{w}^{old} - \eta (r_i -\bar r_i)\mathbf x^{(i)} \quad i: from\quad  1\quad  to\quad m
$$

另一方面，$b$ 的更新方式，可以按 $\partial e /\partial b = 0$ 来得到。

问题看似解决，但上述迭代实际上是不可行的，原因在于所谓的标定距离 $\bar r_i ,i = 1,2,3,,,n$ ，只是一个假设，我们并不能真正知道。但是，上面的分析给我们的启发是，如果知道标定输出和函数输出（分别对应于上面的 $\bar r_i, r_i$ ），那么我们可以利用它们之间的差异来进行参数的更新，而感知器正是如此。

假设存在训练样本集 $S = \{(\mathbf{x}^{(i)}),\,\, y_i)\,|\, i \in \{1,2,,,n\}\}$，其中 $y_i$ 取值为 0 或 1 表示数据的类别，也是判别函数的预期输出或标定输出。

为了方便叙述，下面先把超平面的方程修改一下，原来的方程为

$$
\mathbf{w}^T \mathbf{x} + b = 0
$$

如果定义 $w_0 = 1, x_0 = b$，并且修改 $\mathbf{w} = [w_0, w_1,,w_m],\, \mathbf{x}=[x_0,x_1,,x_m]$。于是得到新的形式

$$
\mathbf{w}^T \mathbf{x} = 0
$$

以及输出函数

$$
r(\mathbf{x}) = \mathbf{w}^T \mathbf{x}
$$

然后定义激活函数

$$
f(x) = \left\{\begin{aligned}
& 0\quad x < 0\\

&1 \quad x >= 0
\end{aligned}
  \right.
$$

判别过程是将激活函数应用于 $\mathbf{w}^T\mathbf{x}$ ， 使得大于等于 0 时，输出 1，否则输出 0。判别过程如下图所示

![](../resources/2017-11-18-neural-networks-perceptron/perceptron.png)

（这里跑一下题，假设我们有一个完美的解决方案（即非常合适的 $f$ 与 $\mathbf{w}$），使得每一个样本特征都能映射到正确的类别，即

$$
f^{-1}\left[\begin{matrix}
y_1\\y_2\\...\\y_n
\end{matrix}\right]=\mathbf{w}^T
\left[\begin{matrix}
\mathbf{x}^{(1)}\\
\mathbf{x}^{(2)}\\...\\
\mathbf{x}^{(n)}
\end{matrix}\right]
$$

然后再定义

$$
Y = [y_1\quad y_2... \quad y_n]^T\\
X = [\mathbf{x}^{(1)T}\quad \mathbf{x}^{(2)T}... \quad \mathbf{x}^{(n)T}]^T
$$

于是可以得到方程

$$
f^{-1}Y = \mathbf{w}^T X
$$

这里的未知量只有 $\mathbf{w}$，于是从某种意义上，相当于在求解一个更具内涵的一元一次方程。跑题完）

为了求解权重向量 $\mathbf{w}$，通过前面的启发我们可以得到 $\mathbf{w}$ 的迭代格式

$$
\mathbf w^{new} = \mathbf{w} - \eta [f(\mathbf{w}^T \mathbf{x}^{(i)}) - y_i]\mathbf x^{(i)}
$$

可以发现，如果 $\mathbf{w}$ 能将特征 $\mathbf x^{(i)}$ 正确分类，即

$$
y_i = f(\mathbf{w}^T x^{(i)})
$$

则有 $\mathbf{w}^{new} = \mathbf{w}$。

end

end

end
