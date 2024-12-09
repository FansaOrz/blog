---
title: CUDA编程基础
date: 2024-12-07 22:27:06
tags: [机器学习, CUDA, C++, 编程语言]
math: true
categories: 编程
excerpt: 介绍基础的CUDA编程
---

# 为什么要用CUDA（GPU和CPU的区别）

- 与CPU相比，GPU（图形处理单元）拥有更多的处理单元（核心），这使得它能够并行处理大量的计算任务。CUDA是NVIDIA推出的一种基于GPU进行通用计算的并行计算平台和编程模型。

- CPU 适合执行逻辑复杂、并行度低的任务

- GPU 适合执行逻辑简单、并行度高的任务

- 以向量加法为例，它逻辑很简单（只需要把两个向量的对应位置加起来即可），且并行度极高（可以同时计算输出向量每个位置上的结果）

    - 如果使用 CPU，需要依次计算输出向量每个位置的结果

    - 如果使用 GPU，我可以同时计算输出向量每个位置的结果，进而大大提高了速度。

## GPU优点

- **高并行性**：GPU包含成百上千个计算核心，能够并行处理大量数据，特别适合处理可以被并行化的计算任务（如矩阵运算、图像处理等）。

- **内存带宽**：GPU的内存带宽远高于CPU，这使得它能够快速处理大规模数据。

- **高吞吐量**：GPU能够在同一时间处理更多的计算任务，从而提高吞吐量。


<p align="center">{% asset_img cpu_gpu.png cpu_gpu %}</p>

# CUDA编程模型

- 在CUDA中，host和device是两个重要的概念，host是指CPU及其内存，device是指GPU及其内存。CUDA程序中既包含host程序，也包含device程序，他们分别在CPU和GPU上运行，同时互相之间可以通信。

- 典型的CUDA程序执行流程如下：

    1. 分配host内存，并进行数据初始化

    2. 分配device内存，并从host内存中拷贝数据到device

    3. 调用CUDA核函数在device上完成指定运算

    4. 从device内存中拷贝数据到host内存

    5. 释放device和host内存

- 上面流程中最重要的一个过程是**调用CUDA的核函数来执行并行计算**，kernel是CUDA中一个重要的概念，是在device上线程中并行执行的函数，核函数用__global__符号声明，在调用时需要用<<<grid, block>>>来指定kernel要执行的线程数量。在CUDA中，每一个线程都要执行核函数，并且每个线程会分配一个唯一的线程号thread ID，这个ID值可以通过核函数的内置变量threadIdx来获得。

- 如何区分host和device上的代码呢？CUDA中通过函数类型限定词来区分，主要的三和函数类型限定词如下：

    - __host__：在host上运行，只能调用host函数，一般省略不写

    - __device__：在device上运行，只能调用device函数，不可以和__global__一起使用

    - __global__：在device上运行，从host中调用，返回类型必须是void，不支持可变参数，不能成为类成员函数。

- grid和block又是什么含义呢？

    - kernel在device上执行时实际上是启动很多县城，一个kernel启动的所有线程，成为一个网格(grid)，同一个grid上的线程共享相同的全局内存空间。

    - grid中包含很多块(block)，每个block中包含很多线程（thread）

    - 因此kernel的执行需要指定grid和block的数量，一般通过dim3来指定。

<p align="center">{% asset_img cuda_model.png Kernel上的两层线程组织结构（2-dim）%}</p>

- 简单CUDA加法程序举例：

    - 线程块大小为(16, 16)，然后将N*N大小的矩阵均分为不同的线程块来执行加法运算

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

# CUDA内存
- CUDA内存层次结构图如下：

<p align="center">{% asset_img cuda_memory.png CUDA内存模型%}</p>

- 全局内存(Global Memory):

    - 全局内存是所有GPU上的线程都可以访问的内存，全局内存中存储了程序中所有的数据

    - 访问速度最慢，但是所有GPU线程都可以访问

- 共享内存(Shared Memory):

    - 共享内存是每个**线程块中所有线程**都可以访问的内存

    - 访问速度介于全局内存和寄存器之间，但是只能被一个线程块中的所有线程访问

- 寄存器内存(Register Memory):

    - 每个线程有自己独立的寄存器内存，访问速度最快

# 参考文档

- [CUDA编程指南](https://www.nvidia.cn/docs/IO/51635/NVIDIA_CUDA_Programming_Guide_1.1_chs.pdf)

- [CUDA编程入门极简教程](https://zhuanlan.zhihu.com/p/34587739)