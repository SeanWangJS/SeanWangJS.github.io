---
layout: post
title: Spark 随笔——特征变换算法扩展
tags: Spark
---

在 spark-mllib 的 org.apache.spark.ml.feature 包下有很多工具能对我们的 dataset 其中一列或几列数据进行变换。比如 StringIndexer 可以按出现频数的顺序对类别特征进行编码，又如 VectorAssembler 能将多列数值特征横向组合成向量特征。虽然 spark-mllib 已经提供了如此多相当便利的工具，但我们时常会有比较特殊的需求，需要定制自己的工具集，值得庆幸的是，Spark 为我们提供了这样的扩展能力。

在 spark-mllib 中，特征转换工具可以被分为两类，在应用的时候可以明显看到区别，第一类工具需要先 fit 整个 dataset，然后才能应用 transform，这是因为此类算法需要知道整个数据集的信息，在 fit 之后得到该算法的一个模型，比如我们常常看到的 StringIndexerModel，并且这些模型可以保存成文件，方便后续加载使用。而第二种工具不依赖整个数据集的信息，可以直接 transform，VectorAssembler 便是其中之一。下面我们介绍本篇文章将要实现的算法。

在 spark-mllib 中的 StringIndexer 能够对一列字符变量进行编码，默认情况下，出现频率最高的字符被置为 0，次之为 1，以此类推。该工具常常能为我们处理类别特征变量带来方便，但是有一个问题是，如果需要编码的类别特征不止一列，那么对每一列特征用 StringIndexer 进行一次 fit 和 transform 则会显得相当低效，因为每次 fit 都需要遍历整个 dataset。于是我们便考虑实现自己的类别特征编码工具，效果和 StringIndexer 一样，但是可以一次性对所有特征列进行变换。直观的讲，假如我们要处理的 dataset 是下面这种结构

|c1|c2|c3|c4|
|:-|:-|:-|:-|
|r1|s1|t1|u3|
|r3|s3|t1|u1|
|r2|s1|t2|u1|
|r1|s2|t3|u2|
|...|...|...|...|

那么首先，我们将每一行 map 成一个哈希表，如下所示

|c1|
|:-|
|((c1, (r1, 1)), (c2, (s1, 1)), (c3, (t1, 1)), (c4, (u3, 1)))|
|((c1, (r3, 1)), (c2, (s3, 1)), (c3, (t1, 1)), (c4, (u1, 1)))|
|((c1, (r2, 1)), (c2, (s1, 1)), (c3, (t2, 1)), (c4, (u1, 1)))|
|((c1, (r1, 1)), (c2, (s2, 1)), (c3, (t3, 1)), (c4, (u2, 1)))|

可以看到，上面的每一行都是一个嵌套的哈希表，最外层表的键为类别名称（比如 c1），值为另一个哈希表，它的键为上述特征名称下面的某个实际量（比如 r1），值为 1。然后我们再对上述 map 后的 dataset 做 reduce，具体逻辑其实就是对哈希表进行合并，内层表遇到相同的键，则值相加，最终结果为

(c1, ((r1, 2), (r3, 1), (r2, 1)))
(c2, ((s1, 2), (s3, 1), (s2, 1)))
(c3, ((t1, 2), (t2, 1), (t3, 1)))
(c4, ((u3, 1), (u1, 2), (u2, 1)))

这也是一个嵌套的哈希表， 外层表的键为类别名称，内层表的键为某个量，值为该量出现的次数。然后我们在分别对每个类别的内层表按出现次数倒排序，比如 
(u3, 1), (u1, 2), (u2, 1)
排序后为
(u1, 2), (u3, 1), (u2, 1)
于是，我们可以将 u1，u3，u2 分别编码为 0，1，2。具体实现如下

```scala
def fit(dataset: Dataset[Row]): MultiColsStringIndexerModel = {

    // 待编码的类别特征列
    val inputColNames = inputCols_

    val spark = dataset.sparkSession
    import spark.implicits._

    val result = dataset
      // map 成嵌套的哈希表
      .map(row => {
        inputColNames.map(col => {
          val value = row.getAs[String](col)
          (col, Map(value -> 1))
        })
          .toMap
    })
      .reduce((m1, m2) => {
        // 嵌套哈希表合并
        inputColNames.map(field => {
          val mm1 = m1(field)
          val mm2 = m2(field)
          val mm3 = (mm1.toList ++ mm2.toList)
            .groupBy(t => t._1)
            .map(t => (t._1, t._2.map(tt => tt._2).sum))
          (field, mm3)
        }).toMap
      })

    val indexers = result.map {
      case (col, valueCount) =>
        val indexer = valueCount.toList
          // 按出现频数倒排序
          .sortWith((t1, t2) => t2._2 - t1._2 < 0)
          .map(t => t._1)
          .zipWithIndex
          .toMap
        (col, indexer)
    }
    new MultiColsStringIndexerModel(indexers, inputCols_, outputCols_)

  }
```

上面的算法给出了 fit 过程，transform 过程则相对简单，只需要通过类别名称和值找到对应的编码即可，但需要注意的是 transform 过程的数据集参数中不能出现 fit 过程没有的类别和值，否则是找不到对应编码的。

```scala
def tranform(dataset: Dataset[Row]): Dataset[Row] = {

    val io = inputColNames.zip(outputColNames).toMap

    val indexers_ = indexers
    var df = dataset
    inputColNames.foreach(icol => {
      val indexer = indexers_(icol)
      df = df.withColumn(io(icol), functions.udf((strVal: String) => {
        indexer(strVal)
      }).apply(functions.col(icol)))
    })

    df

}
```

给出了具体的算法后，我们再来讨论这段代码的组织，按照 spark-mllib 中的风格，fit 过程一般放在算法类中，然后返回模型类，并在模型类中实现 transform 。于是我们的代码结构就可以像下面这样

```scala
class MultiColsStringIndexer{
  def fit(dataset: Dataset[Row]): MultiColsStringIndexerModel = {
     ...
  }
}

class MultiColsStringIndexerModel(val indexers: Map[String, Map[String, Int]], val inputColNames:Array[String], val outputColNames:Array[String]) {
  def tranform(dataset: Dataset[Row]): Dataset[Row] = {
     ...
  }
}
```