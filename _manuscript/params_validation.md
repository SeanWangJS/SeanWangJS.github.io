---
title: 机器学习框架的算法参数校验模块开发
---

#### 引言

有过 scikit-learn 或者其他什么机器学习算法框架使用经验的同学都知道，对于一个模型，它的超参数之间存在依赖关系，某些参数之间的值不能任意组合。比如在 sklearn.linear_model 包下面的 LogisticRegression 模型，它的 penalty, solver, dual 等参数之间就存在一系列的依赖关系，我们可以总结如下
1. 'newton-cg'，'saga'，'lbfgs' 优化仅支持 'l2' 正则参数；
2. 'elasticnet' 正则参数仅可用于 'saga' 优化；
3. 'liblinear' 不能使用 'none' 正则参数，也就是说必须得设置一个正则惩罚项；
4. dual 必须和 'l2' 正则参数以及 'liblinear' 优化一起使用。

当参数设置不满足上述规则的时候，框架会提示错误，比如

```python
>> lr = LogisticRegression(penalty='l1', solver='newton-cg')
>> model=lr.fit(X, y)

>> ValueError: Solver newton-cg supports only 'l2' or 'none' penalties, got l1 penalty.
```