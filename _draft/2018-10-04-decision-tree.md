---
title: 决策树算法
---

信息熵

\[
    H(X) = -\sum_{x\in \chi } p(X = x)\log p(X = x)
    \]

条件熵

\[
    \begin{aligned}
    H(Y|X) &= \sum_{x \in \chi} p(X=x) H(Y|X=x)\\
    &=-\sum_{x\in \chi} p(x) \sum_{y\in \mathcal{Y}}p(y|x)\log p(y|x)\\
    &=-\sum_{x\in \chi }\sum_{y\in \mathcal{Y}} p(x, y) \log p(y|x)
    \end{aligned}
    \]
\[
    p(y|x) = \frac{p(x, y)}{p(x)}
    \]
交叉熵

