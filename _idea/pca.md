---
layout: default
---

考虑一组随机向量

$$
X = \left[
\begin{aligned}
X_1 \\X_2\\...\\...\\X_n
\end{aligned}
\right]
$$

它们的协方差矩阵为

$$
\Sigma = \left[
\begin{aligned}
\sigma_{11}\quad&\sigma_{12}\quad..&\sigma_{1n} \\
\sigma_{21}\quad&\sigma_{22}\quad..&\sigma_{2n}\\
..\quad&..\,\,\,\quad..&..\\
\sigma_{n1}\quad&\sigma_{n2}\quad..&\sigma_{nn}\\
\end{aligned}
\right]
$$

其中 $$\sigma_{ij} = cov(X_i,X_j)$$

定义向量 $$Y$$ 为 $$X$$ 中各元素的线性组合

$$
Y = \sum_{i=1}^n u_i X_i
$$

那么 $$Y$$ 的方差就为

$$
Var(Y) = \sum_{i=1}^n\sum_{j=1}^n u_i u_j \sigma_{ij} = u^T \Sigma u
$$

其中 $$u^T = [u_1 \quad u_2 \quad ... \quad u_n]$$。下面将证明 如果约束 $$u^Tu=1$$， 则$$Var(Y)$$ 的极大值为 $$\Sigma$$ 的最大的特征值。

首先由于 $$\Sigma$$ 为实对称矩阵，所以它拥有一组可以作为单位正交基的特征向量

$$e_1,e_2...e_n$$

满足

$$
e_i^T e_j = \left\{
\begin{aligned}
  1\quad i = j\\0\quad i \ne  j
\end{aligned}\right.
$$

于是可以将 $$u$$ 分解成

$$
u = \sum_{i=1}^n \alpha_i e_i
$$

那么

$$
Var(Y) = \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j e_i^T \Sigma e_j = \sum_{i=1}^n \alpha_i^2 \lambda_i \le \lambda_{max}\sum_{i=1}^n\alpha_i^2
$$

其中 $$\lambda_i$$ 是相对于 $$e_i$$ 的特征值，$$\lambda_{max}$$ 是最大的特征值。然后再考虑约束条件

$$
u^T u=\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j e_i e_j = \sum_{i=1}^n \alpha_i^2 =1
$$

那么显然

$$
Var(Y) \le \lambda_{max}
$$









end
