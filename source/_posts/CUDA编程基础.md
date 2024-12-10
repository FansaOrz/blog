---
title: CUDA编程基础
date: 2024-12-07 22:27:06
tags: [机器学习, CUDA, C++, 编程语言]
math: true
categories: 编程
excerpt: 介绍基础的CUDA编程
---

# 为什么要用 CUDA（GPU 和 CPU 的区别）

- 与 CPU 相比，GPU（图形处理单元）拥有更多的处理单元（核心），这使得它能够并行处理大量的计算任务。CUDA 是 NVIDIA 推出的一种基于 GPU 进行通用计算的并行计算平台和编程模型。

- CPU 适合执行逻辑复杂、并行度低的任务

- GPU 适合执行逻辑简单、并行度高的任务

- 以向量加法为例，它逻辑很简单（只需要把两个向量的对应位置加起来即可），且并行度极高（可以同时计算输出向量每个位置上的结果）

  - 如果使用 CPU，需要依次计算输出向量每个位置的结果

  - 如果使用 GPU，我可以同时计算输出向量每个位置的结果，进而大大提高了速度。

## GPU 优点

- **高并行性**：GPU 包含成百上千个计算核心，能够并行处理大量数据，特别适合处理可以被并行化的计算任务（如矩阵运算、图像处理等）。

- **内存带宽**：GPU 的内存带宽远高于 CPU，这使得它能够快速处理大规模数据。

- **高吞吐量**：GPU 能够在同一时间处理更多的计算任务，从而提高吞吐量。

<p align="center">{% asset_img cpu_gpu.png cpu_gpu %}</p>

# CUDA 编程模型

- 在 CUDA 中，host 和 device 是两个重要的概念，host 是指 CPU 及其内存，device 是指 GPU 及其内存。CUDA 程序中既包含 host 程序，也包含 device 程序，他们分别在 CPU 和 GPU 上运行，同时互相之间可以通信。

- 典型的 CUDA 程序执行流程如下：

  1. 分配 host 内存，并进行数据初始化

  2. 分配 device 内存，并从 host 内存中拷贝数据到 device

  3. 调用 CUDA 核函数在 device 上完成指定运算

  4. 从 device 内存中拷贝数据到 host 内存

  5. 释放 device 和 host 内存

- 上面流程中最重要的一个过程是**调用 CUDA 的核函数来执行并行计算**，kernel 是 CUDA 中一个重要的概念，是在 device 上线程中并行执行的函数，核函数用**global**符号声明，在调用时需要用<<<grid, block>>>来指定 kernel 要执行的线程数量。在 CUDA 中，每一个线程都要执行核函数，并且每个线程会分配一个唯一的线程号 thread ID，这个 ID 值可以通过核函数的内置变量 threadIdx 来获得。

- 如何区分 host 和 device 上的代码呢？CUDA 中通过函数类型限定词来区分，主要的三和函数类型限定词如下：

  - **host**：在 host 上运行，只能调用 host 函数，一般省略不写

  - **device**：在 device 上运行，只能调用 device 函数，不可以和**global**一起使用

  - **global**：在 device 上运行，从 host 中调用，返回类型必须是 void，不支持可变参数，不能成为类成员函数。

- grid 和 block 又是什么含义呢？

  - kernel 在 device 上执行时实际上是启动很多县城，一个 kernel 启动的所有线程，成为一个网格(grid)，同一个 grid 上的线程共享相同的全局内存空间。

  - grid 中包含很多块(block)，每个 block 中包含很多线程（thread）

  - 因此 kernel 的执行需要指定 grid 和 block 的数量，一般通过 dim3 来指定。

<p align="center">{% asset_img cuda_model.png Kernel上的两层线程组织结构（2-dim）%}</p>

- 简单 CUDA 加法程序举例：

  - 线程块大小为(16, 16)，然后将 N\*N 大小的矩阵均分为不同的线程块来执行加法运算

  - 编译指令：nvcc -arch=sm_60 test.cu -o test

```C++
// Kernel定义
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}
int main() {
    ...
    // Kernel 线程配置
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    // kernel调用
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

# CUDA 内存

- CUDA 内存层次结构图如下：

<p align="center">{% asset_img cuda_memory.png CUDA内存模型%}</p>

- 全局内存(Global Memory):

  - 全局内存是所有 GPU 上的线程都可以访问的内存，全局内存中存储了程序中所有的数据

  - 访问速度最慢，但是所有 GPU 线程都可以访问

- 共享内存(Shared Memory):

  - 共享内存是每个**线程块中所有线程**都可以访问的内存

  - 访问速度介于全局内存和寄存器之间，但是只能被一个线程块中的所有线程访问

- 寄存器内存(Register Memory):

  - 每个线程有自己独立的寄存器内存，访问速度最快

# 参考文档

- [CUDA 编程指南](https://www.nvidia.cn/docs/IO/51635/NVIDIA_CUDA_Programming_Guide_1.1_chs.pdf)

- [CUDA 编程入门极简教程](https://zhuanlan.zhihu.com/p/34587739)
