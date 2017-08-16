---
layout: default
---

## 共轭梯度法解线性方程组

之前提到，求解线性方程组 $$A x = b$$ 等价于求下面二次型的极小值

$$
f(x) = \frac 1 2 x^T A x - x^T b
$$

 在最速下降算法中，迭代点沿负梯度方向移动，
 
![](/resources/2017-08-15-equations-solving-conjugate-gradient/contourplot.png)