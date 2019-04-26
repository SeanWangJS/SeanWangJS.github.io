---
layout: post
title: CS231n 可视化识别中的卷积神经网络
---

本文翻译自公开课程[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/linear-classify/?spm=a2c4e.11153940.blogcont167391.30.3c3c235am1A7FH)，感谢原作者。

### 线性分类

在上一节中，我们介绍了图像分类问题，这是一个将图片归类到类别集合中的某一个类别的任务。另外，我们还描述了k邻域（kNN）分类器，该算法通过与训练集中的图片进行比较来给特定图片加标签。正如我们看到的，kNN存在一些缺陷，比如：

* 分类器必须记住所有的训练数据，才能和未来的测试数据进行比较。这相当耗费存储空间，因为这些数据集动辄就是上GB的大小。
* 因为要和所有训练图片进行比较，所以对测试图片进行分类相当耗费。

**概括**. 现在我们准备发展一种更有力的方法用于图片分类问题，并且最终将很自然地延伸到神经网络和卷积神经网络中来。该方法有两个主要的部分：一个得分函数，用于将原始数据映射到类别得分，以及一个损失函数，用来量化预测分数与真实标签之间的相似度。然后，我们将把这转换成一个最优化问题，即针对得分函数中的参数值来最小化损失函数。

### 将图片映射到标签得分

正如前面提到，我们要建立的方法首先要定义得分函数用来将一副图片的像素值映射到针对每一类别的置信得分。当然我们会使用一个具体的例子来建立这一方法。在这之前，让我们假设训练集中的图片 \\(x_i \in  R^D\\)，并且每一张都与一标签 \\(y_i\\) 关联。在这里 \\(i=1...N\\) 且 \\(y_i \in 1...K\\)。也就是说，我们有 **N** 个样本（每个样本的维度为**D**）和 **K** 个不同的类别。例如，在 CIFAR-10 数据集中，我们有 **N** = 50000 张图片，每一张有 **D** = 32 x 32 x 3 = 3072 个像素，并且 **K** = 10，因为有10个不同的类别（例如狗，猫，车等等）。现在我们定义得分函数 \\(f:R^D \rightarrow R^K\\) 将原始图片像素映射到类别得分。

**线性分类器**. 在这一节我们将从基本上算是最简单的函数开始，即线性映射：

\[
f(x_i,W,b) = W x_i + b
\]

在上面的方程中，我们假设将图片 \\(x_i\\) 所有的像素值压平到单个列向量 [Dx1] 中。矩阵 **W** (大小为 [K x D])，以及向量 **b** （大小 [K x 1]）为该函数的**参数**。在CIFAR-10 中，\\(x_i\\) 包含了第i张图片的所有像素值，是一个大小为[3072 x 1]的单列向量，并且 **W** 大小为 [10x3072]，**b** 的大小为 [10x1]，于是3072个数（即原始像素值）被输入到函数中，然后获得10个数（即类别得分）。其中的参数 **W** 常被称为**权重**，而 **b** 常被称为**偏置向量**，因为它会影响输出的分数，却不与 \\(x_i\\) 直接作用。然而，你会常常听到人们将权重和参数两个词换着用。

这里需要注意几件事情：

* 首先，矩阵乘法运算 \\(Wx_i\\) 可以看作是10分类器同时进行分类，其中**W** 的每一行都是一个分类器，于是能并行地高效计算。

* 注意到，我们将数据 \\((x_i,y_i)\\)  当作给定值，并控制参数 **W,b**。我们的目标是通过在整个训练集上匹配分类计算得分与数据的真实标签，来设置参数的值。我们将在后面深入探讨它是怎样起作用的，但是现在，从直观上，我们希望真正的类别相对于其他类别拥有更高的得分。

* 该方法的一大优势是，使用训练数据**W,b**，一旦完成学习，我们就可以丢弃掉整个训练数据集，而只保留学习得到的参数值。之所以能这样做的原因是，一张新的图片可以直接输入分类函数中，并根据计算得分来对其进行分类。

* 最后，注意到，对测试图片进行分类只涉及一个矩阵乘法和向量加法，显然比那种要和整个训练集图片进行比较的算法快得多。

> 事先声明：在卷积神经网络中，图片也会向我们上面介绍的那样被映射成分数值，但是其中的映射函数 (f)会复杂得多，并且包括更多的参数
 
### 对线性分类器的解释

注意到，线性分类器将图片3个通道中的每个像素进行加权求和，并作为类别分数。取决于我们设置的权重，该函数有能力倾向或者厌恶（依赖于权重的符号）图片特定位置的特定颜色。例如，你可以想象一下，如果一幅图的边缘有大面积的蓝色像素（和海水的颜色相近），那么分类器就会更倾向于将图片分类到“船”。于是，你便期望，“船”的分类器在蓝色通道的权重参数中有更多的正数（蓝色的存在会增加船类的得分），并且在红色和绿色通道中，有更多的负值（红绿色的存在会减少船类的得分）。

![](imagemap.jpg)
















