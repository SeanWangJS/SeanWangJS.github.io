---
title: CUDA 编程入门（3）：向量加法示例
tags: CUDA 并行计算
---

前面的文章中，我们简单了解了 GPU 的硬件结构，以及建立在其之上的 CUDA 编程模型，现在，我们来通过一个简单的向量加法例子来展示如何使用 CUDA 的 C++ 扩展来开发一个并行计算程序。

首先，从硬件层面来看，CUDA 程序涉及到 CPU 端和 GPU 端的执行过程，其中 CPU 端又被称为 Host，GPU 端又被称为 Device。一个标准的并行计算流程是：

1. 在 Host 端申请内存空间，分配数据，将数据拷贝到 Device 端；
2. 在 Device 端定义算法，也就是 cuda kernel 方法；
3. 在 Host 端调用 kernel 方法，启动计算任务；
4. 在 Host 端等待计算任务完成，将计算结果从 Device 端拷贝回 Host 端。

下面的代码展示了一个完整的向量加法程序 

```cpp
// main.cu

#include <iostream>
#include <cuda_runtime.h>

// 定义每个block 中的 threads 数量
#define BLOCK_SIZE 256

__global__ void vectorAdd(const float* a, 
                          const float* b,
                          const int n,
                          float* c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {

    // 数组大小
    int n = 1<<20;
    // 计算 block 数量
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 分配 host 内存
    size_t size = n * sizeof(float);
    float* a = (float*)malloc(size);
    float* b = (float*)malloc(size);
    float* c = (float*)malloc(size);
    float* c_ref = (float*)malloc(size); // host 端计算结果，用于验证测试

    // 初始化 host 数组
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    // 分配 device 内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 将 host 数组拷贝到 device 内存
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // 执行 kernel
    vectorAdd<<<numBlocks, BLOCK_SIZE>>>(d_a, d_b, n, d_c);

    // 同步 device
    cudaDeviceSynchronize();

    // 将结果拷贝到 host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // 验证结果
    for (int i = 0; i < n; i++) {
        c_ref[i] = a[i] + b[i];
        if (c[i] != c_ref[i]) {
            printf("Error: c[%d] = %f, c_ref[%d] = %f\n", i, c[i], i, c_ref[i]);
            break;
        }
    }

    // 释放内存
    free(a);
    free(b);
    free(c);
    free(c_ref);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}
```

这段代码有以下几点值得说明的地方：

1. 运行在 Device 端的方法需要加 `__global__` 修饰符，以便编译器能够识别它是一个 kernel 方法；
2. Device 端的内存分配由专门的 CUDA API 来完成，当然，在 CUDA 运行时中还定义了一系列相关方法。
3. kernel 方法调用的标准语法是 `func<<<dimGrid, dimBlock>>>(args..)`，与普通方法不同的地方在于方法名后跟了一个由三个箭头括号包围的参数列表，也就是 `dimGrid` 和 `dimBlock`，这是传输给 CUDA 运行时的参数，用于指定 kernel 方法在设备上的启动配置，其中 `dimGrid` 指定了 block 的数量，`dimBlock` 指定了每个 block 中的 thread 数量，它们都有三个维度，分别对应 x, y, z 轴。在实际使用中，如果设置维度不足 3，则其他维度的值默认为 1，比如 `dimGrid = 10` 的实际含义是 `dimGrid = (10, 1, 1)`。

如果要编译上述代码，需要使用英伟达开发的 nvcc 编译器，nvcc 是在安装 CUDA 时自带的，不需要额外安装。在 Linux 系统中，安装路径在 `/usr/local/cuda/bin`，在 Windows 系统中，默认安装路径在 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v~\bin`。具体的编译命令为

```bash
nvcc main.cu -o main
```

因为 nvcc 编译器已经内置了 CUDA 相关库，所以在编译时不需要指定 CUDA 的头文件和库文件。