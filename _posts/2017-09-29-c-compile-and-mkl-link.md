---
layout: default
---

## C语言编译与 intel MKL 连接

### C语言编译过程

一段c语言源码最终编译成可执行程序要经历多个阶段，包括预处理阶段、编译阶段、汇编阶段以及链接阶段。比如将 hello.c 代码编译成可执行文件将经历如下图所示的过程：

```c
//hello.c
#include <stdio.h>

int main() {
  printf("Hello world");
}
```

![](/resources/2017-09-29-c-compile-and-mkl-link/compile.png)

其中，预处理阶段将源程序使用符号 <span>#</span> 依赖的内容直接复制到代码中，并生成以 <span>.i</span> 为后缀的新文件。对应的 gcc 命令为
```shell
gcc -E hello.c -o hello.i
```
编译阶段将文本文件 hello.i 翻译成汇编语言程序 hello.s ，这也是一个文本文件，其中的内容是汇编代码。

```
gcc -S hello.i -o hello.s
```

汇编阶段将前面的汇编代码翻译成机器语言指令，并打包为 可重定向目标程序，将结果保存在目标文件 hello.o 中。这里的 .o 文件为二进制文件，无法用文本编辑器打开。

```
gcc -c hello.s -o hello.o
```

链接阶段则将本程序与依赖的其他库链接起来生成可执行的程序（在 windows 系统中以 .exe 后缀结尾）。

```
gcc hello.o -o hello
```

上面演示了源程序的每一步编译过程，但是也有一步到位的方法，比如 

```
gcc -c hello.c
```

直接生成 .o 目标文件。甚至可以直接生成可执行文件 

```
gcc hello.c -o hello
```

### 第三方依赖

由 #include 引入的依赖，在预处理阶段，会由编译器进行主动加载。一般对于标准库，加载路径都是默认有效的，但是对于第三方库，编译器并不知道其位置，需要在编译的时候使用 -I 参数手动指定路径。例如如下结构的程序

```
--main.c
--include
  |--add.h
```
其中 main.c 文件内容

```c
//main.c
#include <stdio.h>
#include "add.h"
int main() {
  printf("%d", add(1,1));
}
```

add.h 文件的内容

```c
//add.h
#include <stdio.h>
int add(int a, int b) {
  return a + b;
}
```

上面的 main.c 文件含有对 include 目录下的 add.h 文件依赖，使用引号的 #include 标记，会让编译器首先在 main.c 的所在目录下寻找，如果没有找到，则在由 -I 参数指定的目录下查找。所以这时我们的编译命令为(针对windows系统使用反斜杠)

```shell
gcc -c main.c -I ".\include"
gcc main.o -o main
```

但是这个例子存在一个问题，那就是 .h 后缀的头文件原则上只能有声明，不能有实现，更好的做法是在头文件里写一个求和函数的声明，然后再在 .c 文件中写具体实现（并且 .h 和 .c 文件可以不同名）。比如

```c
// ./include/add.h
int add(int a, int b);
```

```c
// ./include/add.c
int add(int a, int b) {
  return a + b;
}
```

在这种结构下，运行 gcc -c 生成目标 .o 文件时不会出现任何问题，但是在链接 add 函数的时候会发生错误提示：对“add”未定义的引用。原因是编译器找不到 add 函数的实现，虽然它就在头文件的同一个文件夹下，但是编译器哪知道，也并没有义务去主动搜索（万一找到的并不是我们想要的呢）。

### 静态链接库

为了解决刚才的问题，我们可以把 add.h 和 add.c 文件一起编译成库文件供使用者调用。库文件又分为静态链接库和动态链接库，这里先演示静态链接库。

首先在 include 文件夹下编译 add.c

```
gcc -c add.c
```

然后使用 add.o 文件生成静态链接库(在 windows 系统以 lib 后缀结尾)

```
ar r add.lib add.o
```

接着再编译 main.c 生成目标文件 
 
```
gcc -c main.c -I ".\include"
```

最后使用 -L 指定库文件所在目录，以及 -ladd 命令指定我们要调用的库为 add，如果是其他名字的函数，则使用类似 -l{name} 的这种格式。（这其实就表示指定了具体的实现）
 
```
gcc main.o -o main -L .\include -ladd
```

这样即可顺利生成可执行文件。

### 连接英特尔数学核心库

英特尔数学核心库(intel MKL) 是一个高性能的矩阵计算库，实现了诸如 Blas 和 Lapack 等线性代数接口。不少软件（例如 matlab,，numpy等等） 如果以 Blas 和 Lapack 等实现作为底层计算工具，将获得极高的运算速度。

为了连接上 MKL ，上面介绍的知识已经足够了。首先是下载并安装 MKL 库，为了方便指定路径，设置环境变量 

```
%MKL_HOME%=G:\Program Files\IntelSWTools\compilers_and_libraries_2018.0.124\windows\mkl
```

这个路径下包含 include 和 lib 文件夹，提供了编译必要文件。然后下载 [intel官方示例程序](https://software.intel.com/en-us/product-code-samples)，这里以压缩包的 mkl\matrix_multiplication\src 文件夹下的 dgemm_example.c 为例，首先是编译

```
gcc -c dgemm_example.c -I "%MKL_HOME%\include"
```

然后链接静态库，生成可执行文件 dgemm_example.exe 。

```
gcc -o dgemm_example dgemm_example.o -L "%MKL_HOME%\lib\intel64_win" -lmkl_rt
```



