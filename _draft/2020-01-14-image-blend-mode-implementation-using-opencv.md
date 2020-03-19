---
title: 使用 opencv 实现 Photoshop 中的图像混合模式
tags: opencv 图像处理
---

Photoshop 的许多功能还是挺有用的，但是人家毕竟是一个成熟的软件产品，底层接口还是需要自己来实现用起来才方便。刚好有个需求要实现一些滤镜，于是干脆开一个坑，顺便练习一下 opencv 的使用。

每种混合模式都有它的实现原理，我就直接把从网上找的资料列在下面了，A和B都是像素的标准色彩值，介于 0..1 之间，其中 A 是底色，B 是叠加色。

|混合模式|公式|
|--|--|--|
|正常(Normal)| \(B\)|
|正片叠底 (Multiply)|\(A * B\)|
|滤色 (Screen)|\(1 - (1 - A) * (1 - B)\)|

当然上面的所有公式都是在底色和叠加色的不透明度为 100% 的情况下成立，如果底色或者叠加色本身就有一定的透明度，那么就需要额外的修正。我们可以先分析一下最简单的叠加方式，即正常模式，一张不透明度为 \(\alpha_t\) 的图片重叠在一张完全不透明的图片上，混合后的颜色值为 

\[
    r = A (1 - \alpha_t) + B \alpha_t
    \]

可以验证，当 \(\alpha_t = 0\) 时，结果为底色，当 \(\alpha_t=1\) 时，结果为叠加色。

关于这个问题可以参考 stackoverflow: [https://stackoverflow.com/questions/32663192/how-to-reproduce-photoshops-multiply-blending-in-opencv](https://stackoverflow.com/questions/32663192/how-to-reproduce-photoshops-multiply-blending-in-opencv) 下面回答的 GIMP 实现。总的来说，假如底色的不透明度为 \(\alpha_A\)，叠加色的不透明度为 \(\alpha_B\)，并且图像混合时设置的不透明度为 \(c\)，那么混合后图像的综合不透明度计算公式为 

\[
    \alpha = \frac{\min(\alpha_A, \alpha_B) \times c}{\alpha_A + (1 - \alpha_A)  \min(\alpha_A, \alpha_B) \times c}
    \]

