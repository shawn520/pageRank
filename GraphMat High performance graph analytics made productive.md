# GraphMat: High performance graph analytics made productive

# 图矩阵：高性能的图表分析

## ABSTRACT 摘要

Given the growing importance of large-scale graph analytics, there is a need to improve the performance of graph analysis frameworks without compromising on productivity.

鉴于大规模图分析越来越重要，我们需要改进图分析框架的性能而不会影响生产力。

 GraphMat is our solution to bridge this gap between a user-friendly graph analytics framework and native, hand-optimized code. 

我们的解决方案GraphMat弥补了用户友好的图形分析框架和原生的，手工优化的代码之间的差距。

GraphMat functions by taking vertex programs and mapping them to high performance sparse matrix operations in the backend. 

GraphMat函数在后端采取顶点程序并将它们映射到高性能稀疏矩阵。

We thus get the productivity benefits of a vertex programming framework without sacrificing
performance. 

因此我们从顶点编程框架获得生产力的好处，而不会牺牲性能。

GraphMat is a single-node multicore graph frame-work written in C++ which has enabled us to write a diverse set of graph algorithms with the same effort compared to other vertexprogramming frameworks. 

GraphMat是单节点多核图形框架 -与其他顶点相比，具有相同工作量的图算法编程框架，用C ++编写使我们能够编写多样化的集合。

GraphMat performs 1.1-7X faster than high performance frameworks such as GraphLab, CombBLAS and Galois.

与高性能的框架，如GraphLab，CombBLAS和Galois相比，GraphMat执行速度比他们快1.1到1.7倍。

GraphMat also matches the performance of MapGraph, a GPU-based graph framework, despite running on a CPU plat-form with significantly lower compute and bandwidth resources.

尽管在CPU平台上运行，计算和带宽资源显着降低，但GraphMat还与基于GPU的图形框架MapGraph的性能相匹配。

It achieves better multicore scalability (13-15X on 24 cores) than other frameworks and is 1.2X off native, hand-optimized code on a variety of graph algorithms. 

与其他框架相比，它实现了更好的多核可扩展性（24核上的13-15倍），并且在各种图算法上优化了本地手动优化代码的1.2倍。

Since GraphMat performance depends mainly on a few scalable and well-understood sparse matrix opera-tions, GraphMat can naturally benefit from the trend of increasing parallelism in future hardware.

由于GraphMat的性能主要取决于少数可扩展的和易于理解的稀疏矩阵运算，因此GraphMat自然会受益于未来硬件日益增长的并行性趋势。

## INTRODUCTION 引言

Studying relationships among data expressed in the form of graphs has become increasingly important. 

研究以图表形式表达的数据之间的关系变得越来越重要。

Graph processing has become an important component of bio informatics[17], social network analysis[21, 32], traffic engineering[31] etc. 

图形处理已成为生物信息学[17]，社会网络分析[21,32]，交通工程[31]等的重要组成部分。

With graphs getting larger and queries getting more complex, there is a need for graph analysis frameworks to help users extract the information they need with minimal programming effort.

随着图形越来越大，查询变得越来越复杂，图形分析框架需要帮助用户以最少的编程工作来提取他们所需的信息。