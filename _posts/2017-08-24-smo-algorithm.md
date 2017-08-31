---
layout: default
---

SMO 算法

前面对支持向量机的介绍，得到了将要求解的优化问题

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












