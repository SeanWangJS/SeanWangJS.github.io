---
title: 牛顿优化
---

##### 梯度下降迭代优化

在实数邻域无穷可微的函数 \(f(x)\)，其在点 \(x + \Delta x\)处的泰勒展开形式为

\[
  f(x + \Delta x) = f(x) + \frac{\partial f}{\partial x} \Delta x + \frac{\partial^2 f}{\partial x^2} (\Delta x)^2 + ... + \frac{\partial^{n} f}{\partial x^n} (\Delta x)^{(n)}+...
  \]

当 \(\Delta x\) 趋近于 0 时，上式右边每一项都是前一项的高阶无穷小，若我们仅取到一阶精度，那么上式可近似为 

\[
  f(x + \Delta x) \approx f(x) + \frac{\partial f(x)}{\partial x}\Delta x
  \]

