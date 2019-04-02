---
layout: post
title: Hadoop 系列（2）——环境搭建
tags: Hadoop系列
---

Hadoop 的环境搭建对于初学者来讲，确实有点不友好，很多人都是跟着教程一步一步地来，生怕漏掉某一步，到最后却发现还是运行不起来，而且也根本不知道怎么查原因。所以我觉得有必要把配置的方法和原理给记下来，既是一种分享，也算是做笔记。

由于 Hadoop 是用于分布式计算的，但 Jeff Dean 的论文也说了，对于实验性质的项目，最好是在单机上面运行一下，相当于测试。所以我们还是先考虑配置单机环境的 Hadoop。

一般来说，要把 hadoop 运行起来最少需要配置 4 个文件，即 core-site.xml， mapred-site.xml，hdfs-site.xml 和 yarn-site.xml。

在正式配置之前，我们要先了解下 Hadoop 的架构，其实配置时填的那些 xml 文件就是针对 Hadoop 下面那些组件的。

把 Hadoop 下载解压后，目录结构大致如下

.
+-- bin
+-- data
+-- etc
|   +-- hadoop
+-- include
+-- lib
+-- libexec
+-- logs
+-- sbin
+-- share
+-- tmp
+-- LICENSE.txt
+-- NOTICE.txt
+-- README.txt

其中配置文件位于 etc 下的 hadoop 目录，
