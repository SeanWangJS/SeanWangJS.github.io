---
title: 梯度提升算法
---

### 函数估计

训练集 \(\mathcal{S}=\{y_i, \mathbf{x}_i\}_1^N\)

预测函数 \(y = F(\mathbf{x})\)

损失函数 \(L(y_j, F(\mathbf{x}_j)) \quad \quad (y_j, \mathbf{x}_j) \in \mathcal{S}\) 

将 \(y, \mathbf{x}\) 视作随机变量，可得期望损失函数

\[
    E(L(y,F(\mathbf{x}))) = \int\int p(y, \mathbf{x}) L(y, F(\mathbf{x})) \mathrm{d}y\mathrm{d}\mathbf{x}
\]

其中 \(p(y, \mathbf{x})\) 是联合分布的概率密度函数，根据贝叶斯公式

\(p(y, \mathbf{x}) = p(\mathbf{x}) p(y\mid \mathbf{x})\)

可得期望损失

\[
    \begin{aligned}
    E(L(y,F(\mathbf{x}))) &=\int \int p(y, \mathbf{x}) L(y, F(\mathbf{x})) \mathrm{d}y\mathrm{d}\mathbf{x}\\
    &=\int \int p(\mathbf{x}) p(y\mid \mathbf{x}) L(y, F(\mathbf{x})) \mathrm{d}y\mathrm{d}\mathbf{x}\\
    &= \int p(\mathbf{x}) \int p(y\mid \mathbf{x}) L(y, F(\mathbf{x})) \mathrm{d}y\mathrm{d}\mathbf{x} \\
    &=\int p(x) E_y(L(y, F(\mathbf{x})) \mid \mathbf{x}) \mathrm{d}\mathbf{x} \\
    &= E_\mathbf{x} \left(E_y (L(y, F(\mathbf{x}))\mid \mathbf{x})\right)
    \end{aligned} 
    \]

最优预测函数 \(F^*\) 使得期望损失最小化

\[
    F^* = \arg\min_{F} \,  E(L(y,F(\mathbf{x}))) =\arg \min_F \, E_\mathbf{x} \left(E_y (L(y, F(\mathbf{x}))\mid \mathbf{x})\right)
    \]

现假设预测函数的参数向量为 \(\mathbf{P}\)，也就是说 \(y = F(\mathbf{x}; \mathbf{P})\) ，其中 \(\mathbf{P} = \{P_1, P_2, ...\}\) ，并进一步假设预测函数可以写成下述形式

\[
F(\mathbf{x}; \mathbf{P}) = F(\mathbf{x}, \{\beta_i,\mathbf{a}_i \}_1^M) = \sum_{i=1}^M \beta_i h(\mathbf{x}; \mathbf{a}_i)
    \]

### 参数空间上的优化

由于映射 \(y = F(\mathbf{x})\) 取决于参数 \(\mathbf{P}\) ，所以求解最优的 \(F=F^*\) 可以转换为求解最优的 \(\mathbf{P}=\mathbf{P}^*\)

\[
    \mathbf{P}^*=\arg \min \, \Phi(\mathbf{P})
    \]

其中 \(\Phi(P) = E(L(y, F(\mathbf{x};\mathbf{P})))\)

对于迭代优化算法来说，整个过程都是在更新参数向量 \(\mathbf{P}\)，假设其初始值为 \(\mathbf{p}_0\)，后续迭代过程中的每一步增量为 \(\mathbf{p}_i\)。于是，最终的参数值为

\[
    \mathbf{P}^* = \sum_{i=0}\mathbf{p}_i
\]

#### 梯度下降法

梯度下降法是一种迭代方法，每一次迭代都会计算目标函数关于参数向量的梯度

\[
    \mathbf{g}_j = \frac{\partial \Phi (\mathbf{P}_{j-1})}{\partial \mathbf{P}} = \left\{ \frac{\partial \Phi (\mathbf{P}_{j-1})}{\partial P_k} \right\} = \{g_{jk}\}
    \]

设定第\(j\) 次迭代的步长为 \(\rho_j\)，那么该次迭代的参数增量为

\[
    \mathbf{p}_j =- \rho_j \mathbf{g}_j
    \]

### 函数空间上的优化

从另外一个角度来看，我们的目标无非是在具有无限函数的集合中寻找最优的那个函数，这就导致了函数空间的优化问题。在参数空间的优化中，我们要优化的参数是一个向量，每一个下标 \(i\) 都对应着一个值 \(P_i\)，每一次迭代都会对参数向量的值进行更新。类比到函数空间的优化上来，我们可以将函数 \(F(\mathbf{x})\) 看作是一个无限维的参数向量，这时的"下标"其实就是具体的 \(\mathbf{x}\)。于是优化问题就变成了

\[
   F^* = \arg \min_{F} \Phi(F(\mathbf{x}))=\arg \min_F \, E_\mathbf{x} \left(E_y (L(y, F(\mathbf{x}))\mid \mathbf{x})\right)
    \]

令 \(\phi(F(\mathbf{x})) = E_y (L(y, F(\mathbf{x}))\mid \mathbf{x})\)，上式可转换为

\[
    F^* = \arg \min_F \, E_{\mathbf{x}}(\phi(F(\mathbf{x})))
    \]

由于 

\[
    E_{\mathbf{x}}(\phi(F(\mathbf{x}))) = \int p(\mathbf{x}) \phi (F(\mathbf{x})) \mathrm{d}\mathbf{x}
    \]

所以对 \(E_{\mathbf{x}}(\phi(F(\mathbf{x})))\) 的最小化等价于使 \(\phi(F(\mathbf{x}))\) 在每个点 \(\mathbf{x}\) 上取得最小值。

假设初始函数为 \(f_0(x)\)，每次迭代的增量为 \(f_i(\mathbf{x})\)，于是最终的解就为
\[
    F^*(\mathbf{x}) = \sum_{i=0} f_i(\mathbf{x})
    \]

同样地，我们可以计算目标函数对于“参数” \(F(\mathbf{x})\) 的梯度

\[
    g_i(\mathbf{x}) = \frac{\partial \Phi(F(\mathbf{x})_{i-1} )}{\partial F(\mathbf{x})}
    \]

增量函数等于

\[
    f_i(\mathbf{x}) = -\rho_i g_i(\mathbf{x})
    \]






















