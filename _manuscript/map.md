---
title: 对目标检测性能评估指标——平均准确率均值(mAP)——的理解
tags: 深度学习 目标检测 MAP
---

mAP 是评价目标检测模型性能的常用指标，乍一看很不好理解，平均、准确率和均值都是关于什么的呢？下面我们逐一分析。

首先来看相对难理解一点的准确率，对于目标检测任务而言，模型同时给出类别和边界框坐标，对于类别而言，没什么好说的，对就是对，错就是错，但是边界框坐标显然不可能与 Ground Truth 一模一样，所以一般会给一个 IOU 阈值，当 prediction 与 GT 的 IOU 大于该阈值时，则判断为预测准确。


