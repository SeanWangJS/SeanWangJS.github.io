---
layout: post
title: OpenCV 图像处理——非线性拉伸
tags: opencv 图像处理
---

remap 是一个非常有意思的函数，它可以将当前图片按我们给出的规则进行变换，这次我们要研究的是怎样使用 remap 函数进行图片的非均匀拉伸，效果如下图所示

![](/resources/2019-07-15-opencv-image-process-nonlinear-stretch/stretch.png)

可以看到，图片右边的拉伸程度极小，并且越往左，拉伸程度逐渐增大。为了实现这样的效果，我们先简单介绍一下 remap 函数，由于我使用的是 opencv 的 java 绑定库，所以这里以 Imgproc 类中的 remap 作为说明原型

```java
public static void remap(Mat src, Mat dst, Mat map1, Mat map2, int interpolation);
```

该函数有5个参数，其含义分别是

1) src: 原图像 mat，这是我们提供的原始图片。
2) dst: 目标图像 mat。
3) mat1: x 方向映射信息，简单理解就是它以表的形式保存了映射函数。
4) mat2: y 方向映射信息，同 mat1 一样，只不过对应的是 y 方向。
5) interpolation: 插值方式，如果原图相邻的两个像素在映射之后隔了一段距离，那么中间部分像素就需要插值，interpolation 参数便是为插值指定具体方法，比如 Imgproc.INTER_AREA，Imgproc.INTER_LINEAR 等等。

使用 remap 函数最重要的就是构造 mat1 和 mat2 两个矩阵，需要注意的是这两个矩阵的大小应该与 dst 即变换后的图像大小相同。假设 mat1 上位置 \\((i, j)\\) 的元素值为 s， mat2 相同位置上的元素为 t，那么其含义就是把图像 src 上位置 \\((s, t)\\) 的元素映射到 dst 上的 \\((i, j)\\)。为了简单起见，我们先考虑x方向均匀拉伸的情况，此时 y 方向相当于恒等映射，也就是说把 src 的第 1 行映射到 dst 的第 1 行，...，第 n 行映射到第 n 行，于是 mat2 的形式就应该如下所示

$$
    mat2=\left[\begin{aligned}
    0\quad 0 \quad ... \quad 0\\
    1\quad 1 \quad ... \quad 1\\
    ...\\
    n\quad n \quad ... \quad n
    \end{aligned}\right]
$$

而 x 方向上是线性拉伸，假设拉伸 2 倍，那么就是说把 src 上列号为 \\(j\\) 的点映射到 dst 列号为 \\(2j\\) 的位置，我们可以用一条简单的直线来表达这种映射关系

![](/resources/2019-07-15-opencv-image-process-nonlinear-stretch/mapping_linear.png)

于是 mat1 的形式就如下

$$
    mat1 = \left[\begin{aligned}
    0\quad 0 \quad ... \quad n \quad n\\
    0\quad 0 \quad ... \quad n \quad n\\
    ...\\
    0\quad 0 \quad ... \quad n \quad n
    \end{aligned}\right]
    $$

使用上面两个矩阵，我们可以得到下图

![](/resources/2019-07-15-opencv-image-process-nonlinear-stretch/stretch_linear.jpg)

显然符和条件。接下来我们考虑非线性拉伸，注意我们的要求是图片最右端的拉伸程度为 0 ，最左端拉伸程度达到最大，这时映射关系应该是一条曲线，并且最右端的切线斜率应该等于 1。假设总体拉伸仍是 2 倍，那么映射曲线如下

![](/resources/2019-07-15-opencv-image-process-nonlinear-stretch/mapping_unlinear.png)

当然，满足这一条件的曲线很多，我们这里就以二次曲线来做说明，因为已知两个点的坐标，以及一个点上的切线斜率可以计算出二次曲线的函数。比如本例中该曲线的函数形式为

$$
    f(x) = \frac 1 {4n} x^2
    $$

用此公式填充 mat1 矩阵便可得到最初图片所示的效果。当然除此之外我们还可以使用更多的公式实现更复杂的效果，比如单个周期的正弦波在两个方向映射可以获得两个方向的对称效果。

$$
    \begin{aligned}
    f(x) = \sin(\frac  x n)\\
    f(y) = \sin(\frac  y n)
    \end{aligned}
    $$

![](/resources/2019-07-15-opencv-image-process-nonlinear-stretch/sin.jpg)