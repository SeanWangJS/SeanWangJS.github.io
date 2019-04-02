---
title: Gradient Boosting Machine
---

In a predictive problem, we need to give an output value, say \\(y\\), from a known variable \\(\mathbf{x}\\). In other words, we need a function \\(y=F(\mathbf{x})\\) which is, however, unknown yet. The only thing we know is a set of samples \\(\mathcal{S}=\{\mathbf{x}_i, y_i \mid i =1,2,,,N\}\\) where \\(\mathbf{x}_i\\) and \\(y_i\\) are corresponded to each other.

So our goal is trying to find a perfect function \\(F(\mathbf{x})\\) which map every \\(\mathbf{x}_i\\) to \\(y_i\\), i.e.

\[
    y_i = F(\mathbf{x}_i) \quad i = 1,2,,,N
    \]

