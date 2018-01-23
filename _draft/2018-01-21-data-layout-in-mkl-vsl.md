---
layout: post
title: MKL 向量统计库 VSL 中的数据索引问题
---

由于 MKL 支持 C 与 Fortran 两种数据索引风格，在使用 VSL 进行统计计算的时候，可以使用参数 VSL_SS_MATRIX_STORAGE_ROWS 和 VSL_SS_MATRIX_STORAGE_COLS 来指定我们传入的数据布局为行索引或者列索引。但是我们一般习惯的是类 C 语言的行索引风格，所以这篇文章就来讨论一下如何使用 VSL 这两种风格的适当组合来适配矩阵统计量的计算。

### C 风格与 Fortran 风格的数据索引方式

首先来看一下什么是 C 风格与 Fortran 风格的数据索引。内存可以被认为是一种一维结构，只需要知道一个索引号就能取出里面的数据，但实际问题中很多数据都是二维的，比如矩阵存储的数据。为了将现实问题中的数据存到内存里面，以及从内存里面提取数据，我们需要对两者建立映射关系，这个映射公式就是区分两种索引风格的依据，比如这段数据

![](/resources/2018-01-21-data-layout-in-mkl-vsl/raw.png)

假设它代表的实际数组是下面这样的

![](/resources/2018-01-21-data-layout-in-mkl-vsl/row.png)

这个二维数组索引 (i,j) 映射到内存块中的索引为 i * 5 + j，其中的 5 是子数组的长度，而这种索引风格就是类 C 的。而同样的数组如果用 Fortran 风格的索引来表示，则相当于把 C 风格的数组给转置一下

![](/resources/2018-01-21-data-layout-in-mkl-vsl/column.png)

这时二维数组索引 (i,j) 上的数据在内存中的索引为 j * 5 + i。值得注意的是，数组的索引风格不取决于它的形状是横条或者竖条的，而是它的索引映射公式，或者观察它的二维数组视图和内存布局。如果内存中的数据遍历结果与二维数组视图中的横向遍历结果相同，那么就是 C 风格索引，而如果内存中的数据遍历结果与二维数组视图中的竖向遍历结果相同，则是 Fortran 风格的。

为了进一步说明，我们考虑鸢尾花数据集的前 3 个样本

![](/resources/2018-01-21-data-layout-in-mkl-vsl/iris.png)

从这张图我们可以看到，样本的维度 DIM = 4，样本数量 N = 3。但仅通过这个二维数组视图我们并不能判断它是 C 或者 Fortran 风格的，因为两者皆允，若我们认为里面的数据是 C 风格索引的，则在内存中是下面这样的

![](/resources/2018-01-21-data-layout-in-mkl-vsl/iris-c.png)

而如果按 Fortran 的列存储风格，则内存中的数据布局如下所示

![](/resources/2018-01-21-data-layout-in-mkl-vsl/iris-f.png)

### 使用 VSL 统计计算

向量统计库设计了一套完整的任务流程API，大致分成三个阶段：新建任务，编辑任务以及计算，分别对应于三个函数 vsl\*NewTask, vsl\*EditTask，vsl*SSCompute 其中的 \* 号可以取 d s i 分别代表双精度浮点数、单精度浮点数以及整数。

现在考虑使用 VSL 计算前面给的鸢尾花数据的一些统计值，来探讨它的内部是如何处理数据布局的，首先来看看每个维度上的均值，代码如下

```c
//main.c
#include <stdio.h>
#include "mkl.h"

int main()
{
    /* 数组定义 \*/
	  double matrix[] = {5.1, 3.5, 1.4, 0.2, 4.9, 3.1, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2};
	  MKL_INT DIM = 4;
    MKL_INT N = 3;
    MKL_INT x_storage = VSL_SS_MATRIX_STORAGE_COLS; //使用 Fortran 的列索引风格

    VSLSSTaskPtr task;
    double mean[DIM];
    int errcode;
    unsigned MKL_INT64 estimate = VSL_SS_MEAN;

    // 新建任务：导入数组描述
    errcode = vsldSSNewTask( &task, &DIM, &N, &x_storage, (double*)matrix, 0, 0 );
    // 编辑任务：设定任务为均值计算
    errcode = vsldSSEditTask( task, VSL_SS_ED_MEAN, mean);
    // 开始计算任务
    errcode = vsldSSCompute( task, estimate, VSL_SS_METHOD_FAST );

	  printf("%f,%f,%f,%f\n",mean[0], mean[1],mean[2],mean[3]);

    // 释放资源
    errcode = vslSSDeleteTask( &task );
    MKL_Free_Buffers();
    return errcode;
}
/*
compile command:
gcc -c main.c -o target.o -I "%MKL_ROOT%\include"
gcc -o target.exe target.o -L "%MKL_ROOT%\lib\intel64_win" -lmkl_rt
*/
//4.900000,3.266667,1.366667,0.200000
```

代码中的 matrix[] 数组就是数据在内存中的存储方式，仅需要一个索引号即可定位数据。然后通过指定数据的存储风格为列存储

```c
MKL_INT x_storage = VSL_SS_MATRIX_STORAGE_COLS;
```

此时数据的二维视图如下图所示

![](/resources/2018-01-21-data-layout-in-mkl-vsl/iris-f2.png)

然后指定样本的维度 DIM = 4， 数量 N = 3，最后正确计算出了每个维度上的均值。

我们可以猜测一下 VSL 内部是如何进行计算的，首先从第 0 个元素开始，以 DIM 为间隔跳跃 N - 1 次，对所到数据进行求和，最后求平均

![](/resources/2018-01-21-data-layout-in-mkl-vsl/mean1.png)

然后从第 1 个元素开始，做相同的计算

![](/resources/2018-01-21-data-layout-in-mkl-vsl/mean2.png)

直到达到第 DIM 个元素

![](/resources/2018-01-21-data-layout-in-mkl-vsl/meann.png)

最后将所求均值填充到数组 mean 所对应的内存。

如果我们在指定数组存储风格的时候，将代码替换为

```c
MKL_INT x_storage = VSL_SS_MATRIX_STORAGE_ROWS;
```

这时数组的二维视图如下

![](/resources/2018-01-21-data-layout-in-mkl-vsl/iris.png)

如果其他代码不变，我们将得到如下结果

$$
3.333333,2.733333,2.100000,1.566667
$$

这一结果有些意外，但是我们仍能解释，它的计算过程其实就是对连续 N 个数取平均，并重复 DIM 次

![](/resources/2018-01-21-data-layout-in-mkl-vsl/meanr.png)

从上面的两次计算可以看到，如果在指定数据存储风格时不够严谨，得到的结果可能出乎我们的意料。

但是我们也能从上面的计算中总结一些规律，我们看到，无论指定的存储风格是 类 C 的行存储，还是 Fortran 的列存储。VSL 内部都是在二维数组视图的行方向上计算均值，并且每次都是取 N 个数，共计算 DIM 次。

![](/resources/2018-01-21-data-layout-in-mkl-vsl/iris-f2-draw.png)

![](/resources/2018-01-21-data-layout-in-mkl-vsl/iris-draw.png)

掌握了这一规律之后，我们就能继续向我们的目标迈进一步，即通过维度 DIM，数量 N，以及存储风格的变换来适配矩阵的按行或者按列的统计量的计算。

### 计算矩阵的统计量

假设我们的矩阵是这样的

$$
\left[
\begin{aligned}
5.1\quad\quad& 4.9& 4.7\\
3.5\quad\quad& 3.1& 3.2\\
1.4\quad\quad& 1.4& 1.3\\
0.2\quad\quad& 0.2& 0.2
\end{aligned}
\right]
$$

它的一维表示形式为

$$
[5.1\quad 4.9\quad 4.7\quad
3.5\quad 3.1\quad 3.2\quad
1.4\quad 1.4\quad 1.3\quad
0.2\quad 0.2\quad 0.2]
$$

如果要计算每行的均值，根据前面的讨论，我们只需要设置 DIM = 4，N = 3，x_storage = VSL_SS_MATRIX_STORAGE_COLS。但如果要计算每列的均值，那么仅仅是设置 x_storage = VSL_SS_MATRIX_STORAGE_ROWS 是不够的，因为 N = 3，于是每次读取了 3 个数，总共计算 DIM = 4 次。而如果要按列计算，应该每次读 4 个数，总共计算 3 次，也就是说还应该交换 DIM 与 N 的值。修改后代码如下

```c
#include <stdio.h>
#include "mkl.h"

int main()
{
	  double matrix[] = {5.1, 3.5, 1.4, 0.2, 4.9, 3.1, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2};
	  MKL_INT DIM = 3;
    MKL_INT N = 4;

    VSLSSTaskPtr task;
    MKL_INT x_storage = VSL_SS_MATRIX_STORAGE_ROWS;
    double mean[DIM];
    int errcode;
    unsigned MKL_INT64 estimate = VSL_SS_MEAN;

    errcode = vsldSSNewTask( &task, &DIM, &N, &x_storage, (double*)matrix, 0, 0 );
    errcode = vsldSSEditTask( task, VSL_SS_ED_MEAN, mean);
    errcode = vsldSSCompute( task, estimate, VSL_SS_METHOD_FAST );

	  printf("%f,%f,%f\n",mean[0], mean[1],mean[2]);

    errcode = vslSSDeleteTask( &task );
    MKL_Free_Buffers();
    return errcode;
}
//output: 2.550000,2.400000,2.350000
```

但其实对于矩阵来说，DIM 和 N 这两个符号的意义并不明显，因为我们一般使用行数或者列数来描述矩阵的形状。所以直接改成行数 ROW 和列数 COL 应该更加合适一点。但还有一个问题是应该是将 DIM 改成 ROW，N 改成 COL？还是反过来？

这一问题也好解决，在前面当 x_storage = VSL_SS_MATRIX_STORAGE_COLS 时，DIM = 4，N = 3，所以 DIM 就是行数，N 就是列数。

接下来我们再考虑写一个通用的函数，来简化按行按列这两种情况的计算。它的参数应该被精心设计，首先应该能传递矩阵的数据，也就是一个数组指针，然后还应该让我们知道矩阵的行列数，以及一个标志用来表示我们需要按行或者按列计算，当然还有一个指针用来存储计算结果。它看起来应该是下面这样的

```c
int mean(double* matrix, int ROW, int COL, int axis, double* m);
```

具体函数的代码如下

```c
int mean(double* matrix, int ROW, int COL, int axis, double* m) {
	  VSLSSTaskPtr task;
    MKL_INT x_storage = (axis == 0)?VSL_SS_MATRIX_STORAGE_COLS:VSL_SS_MATRIX_STORAGE_ROWS;
	  if(axis != 0) {
		   int temp = ROW;
		   ROW = COL;
		   COL = temp;
	  }
	  int errcode;
    unsigned MKL_INT64 estimate = VSL_SS_MEAN;

	  errcode = vsldSSNewTask( &task, &ROW, &COL, &x_storage, (double*)matrix, 0, 0 );
    errcode = vsldSSEditTask( task, VSL_SS_ED_MEAN, m);
    errcode = vsldSSCompute( task, estimate, VSL_SS_METHOD_FAST );

    errcode = vslSSDeleteTask( &task );
    MKL_Free_Buffers();
    return errcode;
}
```

我们定义矩阵的时候按照实际情况设置它的 ROW 和 COL

```c
double matrix[] = {5.1, 3.5, 1.4, 0.2, 4.9, 3.1, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2};
int ROW = 4;
int COL = 3;
double m[ROW];
```

 然后调用函数，如果要计算每行的均值，令 axis = 0，否则计算每列的均值。

 ```c
mean(matrix, ROW, COL, 0, m);
printf("%f,%f,%f,%f\n",m[0], m[1],m[2],m[3]);
// output: 4.900000,3.266667,1.366667,0.200000

mean(matrix, ROW, COL, 1, m);
printf("%f,%f,%f\n",m[0], m[1],m[2]);
// output: 2.550000,2.400000,2.350000
 ```

### 总结

关于数组存储风格的讨论会稍微有点绕，因为这里面很多参数都在变动，比如二维数组的视图，索引类型，还有矩阵行列与样本数据的维度和数量的关系等等。经过本篇文章的讨论，我们看到只需要抓住最关键的部分，情况会清晰不少。也就是说，对于一个矩阵，我们可以按行连续的方式把它存到一维数组，在调用 VSL 代码时，如果要按行计算，只用设定存储方式为 VSL_SS_MATRIX_STORAGE_COLS，而若要按列计算，则除了要将存储方式设为 VSL_SS_MATRIX_STORAGE_ROWS 外，还需要交换 ROW 与 COL 的值。
