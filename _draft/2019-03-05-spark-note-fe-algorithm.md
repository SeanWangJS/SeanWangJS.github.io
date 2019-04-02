---
title: Spark 随笔——特征变换算法扩展
---

在 spark-mllib 的 org.apache.spark.ml.feature 包下有很多工具能对我们的 dataset 其中一列或几列数据进行变换。比如 StringIndexer 可以按出现频数的顺序对类别特征进行编码，又如 VectorAssembler 能将多列数值特征横向组合成向量特征。虽然 spark-mllib 已经提供了如此多相当便利的工具，但我们时常会有比较特殊的需求，需要定制自己的工具集，值得庆幸的是，Spark 为我们提供了这样的扩展能力。

在 spark-mllib 中，特征转换工具可以被分为两类，在应用的时候可以明显看到区别，第一类工具需要先 fit 整个 dataset，然后才能应用 transform，这是因为此类算法需要知道整个数据集的信息，在 fit 之后得到该算法的一个模型，比如我们常常看到的 StringIndexerModel，并且这些模型可以保存成文件，方便后续加载使用。而第二种工具不依赖整个数据集的信息，可以直接 transform。

