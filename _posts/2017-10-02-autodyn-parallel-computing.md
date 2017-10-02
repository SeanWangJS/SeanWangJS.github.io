---
layout: default
---

## AUTODYN 并行计算（单机）

使用多核 CPU 来加速 AUTODYN 计算是一个行之有效的方法，而且设置也相当简单，这里稍微记录一下。

首先是 MPI 路径，在安装 ANSYS 的时候就已经装上了 MPI，可以在 ANSYS 主目录下的 commonfiles 文件夹下找到 ，这里要做的是将其路径加入系统环境变量，在 cmd 中运行 mpirun 命令应该能出现一系列提示。

然后是编写脚本 ，在任一文件夹下创建 applfile 文件（无扩展名），加入一段参数

```
-e MPI_FLAGS=y0 -e ANSYS_EXD_MPI_TYPE=pcmpi -h machine1 -np 1 "G:\Program Files\ANSYS Inc\v162\aisol\AUTODYN\winx64\autodyn.exe"
-h machine1 -np 2 "G:\Program Files\ANSYS Inc\v162\aisol\AUTODYN\winx64\adslave.exe"
```

意思是在 machine1 机器上启动了一个主节点，以及两个从节点。machine1 应该在 hosts 文件中填写

```
127.0.0.1 machine1
```

表示 machine1 就是本机，否则会出现 gethostname 错误。

需要注意的是，第一个 -h 前面一定不要换行，不然会出现 MPI 找不到程序的错误，困扰了好久。

然后通过 MPI 启动 AUTODYN

```
mpirun  -prot -e MPI_WORKDIR="G:\Program Files\ANSYS Inc\v162\AISOL\AUTODYN\winx64" -f applfile
```

当然这段命令应该在 applfile 所在文件夹下运行。启动之后按正常方式使用 AUTODYN，但在运行之前还应给每个核心设置并行计算任务，在 AUTODYN 的 parallel 标签下，也很简单，看看就会了（但是设置是否合理关系到计算效果，一般来说，应该给每个核心分配大致相等的计算量）。