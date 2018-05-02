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

There has been an explosion of graph programming frameworks in recent years [1, 3, 4, 5, 15, 19, 30]. 

近年来图形编程框架发展迅猛[1,3,4,5,15,19,30]。

All of them claim to provide good productivity, performance and scalability. 

他们都声称提供良好的生产力，性能和可扩展性。

However, a recent study has shown [28] that the performance of most frameworks is off by an order of magnitude when compared to native, hand-optimized code. 

然而，最近的一项研究表明[28]，与原生的，手工优化的代码相比，大多数框架的性能都要下降一个数量级。

Given that much of this performance gap remains even when running frameworks on a single node [28], it is imperative to maximize the efficiency of graph frameworks on existing hardware (in addition to focusing on scale out issues). 

考虑到即使在单个节点上运行框架，这种性能差距仍然存在[28]，所以必须最大限度地提高现有硬件上的图形框架的效率（除了侧重于扩展问题外）。

GraphMat is our solution to bridge this performance-productivity gap in graph analytics.

我们的解决方案GraphMat可以弥合图形分析中的性能与生产力差距。

The main idea of GraphMat is to take vertex programs and map them to generalized sparse matrix vector multiplication operations. 

GraphMat的主要思想是获取顶点程序并将它们映射到广义稀疏矩阵向量乘法运算。

We get the productivity benefits of vertex programming while enjoying the high performance of a matrix backend. 

在享受矩阵后端的高性能的同时，我们获得顶点编程的生产力优势。

In addition, it is easy to understand and reason about, while letting users with knowledge of vertex programming a smooth transition to a high performance environment. 

此外，很容易让用户理解和推理顶点编程顺利过渡到高性能环境。

Although other graph frameworks based on matrix operations exist (e.g. CombBLAS [3] and PEGASUS
[19]), GraphMat wins out in terms of both productivity and performance as GraphMat is faster and does not expose users to the underlying matrix primitives (unlike CombBLAS and PEGASUS).

尽管存在其他基于矩阵运算的图框架（例如CombBLAS [3]和PEGASUS）[19]），GraphMat在生产力和性能两方面均胜出，因为GraphMat速度更快，并且不会将用户暴露于底层矩阵基元（与CombBLAS和PEGASUS不同）。

We have been able to write multiple graph algorithms in GraphMat with the same effort as other vertex programming frameworks. Our contributions are as follows:

我们已经能够使用与其他顶点编程框架相同的工作量在GraphMat中编写多个图形算法。 我们的贡献如下：

1. GraphMat is the first multi-core optimized vertex programming model to achieve within 1.2X of native, hand-coded, optimized code on a variety of different graph algorithms.

   GraphMat是第一个多核优化的顶点编程模型，可在各种不同的图形算法的本地手动编码优化代码的1.2倍以内实现。

   GraphMat is 5-7X faster than GraphLab [5] & CombBLAS and 1.1X faster than Galois [4] on a single node. 

   在单个节点上，GraphMat比GraphLab [5]和CombBLAS快5-7倍，比Galois [4]快1.1倍。

   It also matches the performance of MapGraph [15], a recent GPU-based graph framework running on a contemporary GPU.

   它也与MapGraph [15]的性能相匹配，这是最近在当代GPU上运行的基于GPU的图形框架。

2. GraphMat achieves good multicore scalability, getting a 13-15X speedup over a single threaded implementation on 24 cores. 

   GraphMat实现了良好的多核可扩展性，通过24核上的单线程实现获得13-15倍的加速比。

   In comparison, GraphLab, CombBLAS, and Galois scale by only 2-12X over their corresponding single threaded implementations.

   相比之下，GraphLab，CombBLAS和Galois仅比其相应的单线程实现缩小了2-12倍。

3. GraphMat is productive for both framework users and developers. 

   GraphMat对于框架用户和开发人员都很有成效。

   Users do not have to learn a new programming paradigm(most are familiar with vertex programming), where as backend developers have fewer primitives to optimize as it is based on Sparse matrix algebra, which is a well-studied operation in High Performance Computing (HPC) [35].

   用户不需要学习新的编程范例（大部分人都熟悉顶点编程），后端开发人员基于稀疏矩阵代数进行优化的基元更少，因为它是高性能计算（HPC）中深入研究的操作 ）[35]。

Matrices are fast becoming one of the key data structures for databases, with systems such as SciDB [6] and other array stores becoming more popular. 

矩阵正在迅速成为数据库的关键数据结构之一，诸如SciDB [6]和其他数组商店等系统越来越受欢迎。

Our approach to graph analytics can take advantage of these developments, letting us deal with graphs as special cases of sparse matrices. 

我们的图分析方法可以利用这些发展，让我们将图作为稀疏矩阵的特例处理。

Such systems offer transactional support, concurrency control, fault tolerance etc. 

这样的系统提供事务支持，并发控制，容错等。

while still maintaining a matrix abstraction. We offer a path for array processing systems to support graph analytics through popular vertex programming frontends.

同时仍然保持矩阵抽象。 我们为阵列处理系统提供了一条途径，通过流行的顶点编程前端支持图形分析。

Basing graph analytics engines on generalized sparse matrix vector multiplication (SPMV) has other benefits as well. 

基于广义稀疏矩阵向量乘法（SPMV）的图分析引擎也具有其他优点。

We can leverage decades of research on techniques to optimize sparse linear algebra in the High Performance Computing world. 

我们可以利用数十年的技术研究来优化高性能计算领域的稀疏线性代数。

Sparse linear algebra provides a bridge between Big Data graph analytics and High Performance Computing. 

稀疏线性代数提供了大数据图形分析和高性能计算之间的桥梁。

Other efforts like GraphBLAS [23] are also part of this growing effort to leverage lessons learned from HPC to help big data.

像GraphBLAS [23]这样的其他工作也是这项不断增长的努力的一部分，这些努力将HPC的经验教训用于帮助大数据。

The rest of the paper is organized as follows. 

本文的其余部分安排如下。

Section 2 provides motivation for GraphMat and compares it to other graph frame-works. 

第2部分为GraphMat提供了动机，并将其与其他图形框架进行了比较。

Section 3 discusses the graph algorithms used in the paper. 

第3节讨论了本文中使用的图算法。

Section 4 describes the GraphMat methodology in detail. 

第4节详细介绍了GraphMat方法。

Section 5 gives details of the results of our experiments with GraphMat while Section 6 concludes the paper.

第5部分详细介绍了我们用GraphMat进行实验的结果，第6部分结束了本文。

## 2. MOTIVATION AND RELATED WORK 动机和相关工作

Graph analytics frameworks come with a variety of different programming models. 

图分析框架带有各种不同的编程模型。

Some common ones are vertex programming(“think like a vertex”), matrix operations (“graphs are sparse matrices”), task models(“vertex/edge updates can be modeled as tasks”),declarative programming (“graph operations can be written as data-log programs”), and domain-specific languages (“graph processing needs its own language”). 

一些常见的是顶点编程（“像一个顶点思考”），矩阵运算（“图形是稀疏矩阵”），任务模型（“顶点/边缘更新可以建模为任务”），声明式编程（“图形操作可以 写成数据日志程序“）和特定于领域的语言（”图形处理需要它自己的语言“）。

Of all these models, vertex programming has been quite popular due to ease of use and the wide variety of different frameworks supporting it [28].

在所有这些模型中，由于易用性和支持它的各种不同框架，顶点编程已经非常流行[28]。

## 3. ALGORITHMS 算法

To showcase the performance and productivity of GraphMat, we picked five different algorithms from a diverse set of applications, including machine learning, graph traversal and graph statistics. 

为了展示GraphMat的性能和生产力，我们从多种应用中选择了五种不同的算法，包括机器学习，图形遍历和图形统计。

Our choice covers a wide range of varying functionality (e.g.traversal or statistics), data per vertex, amount of communication,iterative vs. non iterative etc. 

我们的选择涵盖了各种各样的功能（例如，遍历或统计），每个顶点的数据，通信量，迭代与非迭代等。

We give a brief summary of the algorithms below.

我们给出下面算法的简要总结。

###  3.1 Page Rank (PR) 网页排名算法

This is an iterative algorithm used to rank web pages based on some metric (e.g. popularity). 

这是一种迭代算法，用于根据某种度量（例如流行度）对网页进行排名。

The idea is compute the probability that a random walk through the hyperlinks (edges) would end in a particular page (vertex). 

这个想法是计算一个随机遍历超链接（边）将在特定页面（顶点）中结束的概率。

The algorithm iteratively updates the rank of each vertex according to the following equation:

该算法根据以下等式迭代地更新每个顶点的等级：
$$
PR^{t+1}(v) = r + (1-r) * \sum_{u|(u,v) \in E)}\frac{PR^t(u)}{degree(u)}
$$
where PR t (v) denotes the page rank of vertex v at iteration t,E is the set of edges in a directed graph, and r is the probability of random surfing. The initial ranks are set to 1.0.

其中PR t（v）表示迭代t处顶点v的页面排名，E是有向图中的边集，r是概率随机冲浪。 初始等级设置为1.0。

### 3.2 Breadth First Search (BFS) 广度优先搜索算法

This is a very popular graph search algorithm, which is also used as the kernel by the Graph500
benchmark [24]. 

这是一种非常流行的图形搜索算法，它也被Graph500用作内核基准[24]。

The algorithm begins at a given vertex (called root)and iteratively explores all connected vertices of an undirected and unweighted graph. 

该算法从给定顶点（称为根）开始，并迭代地探索无向图和未加权图的所有连接顶点。

The idea is to assign a distance to each vertex, where the distance represents the minimum number of edges needed to be traversed to reach the vertex from the root. 

这个想法是给每个顶点分配一个距离，其中距离表示从根到达顶点需要遍历的最小边数。

Initially, the distance of the root is set to 0 and it is marked active. The other distances are set to infinity. 

最初，根的距离设置为0，并且它被标记为活动。其他距离设置为无穷大。 

At iteration t, each vertex adjacent to an active vertex computes the following:

迭代t时，与活动顶点相邻的每个顶点计算以下内容：
$$
Distance(v) = min(Distance(v),t + 1)
$$
If the update leads to a change in distance (from infinity to t+1),then the vertex becomes active for the next iteration.

如果更新导致距离（从无穷远到t + 1）的变化，则顶点在下一次迭代中变为活动状态。

### 3.3 Collaborative Filtering (CF) 协作过滤算法

This is a machine learning algorithm used by many recommender systems [27] for estimating a user’s rating for a given item based on an incomplete set of (user, item) ratings. 

这是一种由许多推荐系统[27]使用的机器学习算法，用于基于（用户，项目）评分的不完整集合来估计给定项目的用户评分。

The underlying assumption is that users’ ratings are based on a set of hidden/latent features and each item can be expressed as a combination of these features. 

潜在的假设是用户的评分基于一组隐藏/潜在特征，每个项目可以表示为这些特征的组合。

Ratings depend on how well the user’s and item’s features match. 

评分取决于用户和项目的功能匹配程度。

Given a matrix Gof ratings, the goal of collaborative filtering technique is to compute two factors P U and P V , each one is a low-dimensional dense matrix. 

给定一个矩阵Gof评分，协同过滤技术的目标是计算两个因子P U和P V，每个因子是一个低维密集矩阵。

This can be accomplished using incomplete matrix factorization [20]. 

这可以通过使用不完全矩阵分解来实现[20]。

Mathematically, the problem can be expressed as eq. (3) 

在数学上，这个问题可以表示为等式。（3）

where u and v are the indices of the users and items, respectively, G uv is the rating of the u th user for the v th item, p u &p v are dense vectors of length K corresponding to each user and item, respectively.

其中u和v分别是用户和项目的指数，G uv是第v个项目的第u个用户的评分，p u和p v分别是对应于每个用户和项目的长度K的密集向量。

Matrix factorization is usually performed iteratively using Stochastic Gradient Descent (SGD) or Gradient Descent (GD). 

矩阵分解通常使用随机梯度下降（SGD）或梯度下降（GD）迭代执行。

In each iteration t, GD performs Equation 4 - 6 for all users and items. 

在每次迭代t中，GD对所有用户和项目执行公式4-6。

SGD performs the same updates without the summation in equation 5 on all ratings in a random order. 

SGD在没有等式5中的所有评级的总和的情况下以随机顺序执行相同的更新。

The main difference between GD and SGD is that GD updates all the p u and p v once per iteration instead of once per rating as in SGD.

GD和SGD之间的主要区别在于，GD每次迭代更新所有的p和p，而不是像SGD中那样每次更新一次。

### 3.4 Triangle Counting (TC) 三角计数

 This is a statistics algorithm useful for understanding social networks, graph analysis and computing clustering coefficient. 

这是一个统计算法，用于理解社交网络，图形分析和计算聚类系数。

The algorithm computes the number of triangles in a given graph. 

该算法计算给定图形中的三角形数量。

A triangle exists when a vertex has two adjacent vertices that are also adjacent to each other. 

当一个顶点有两个相邻的顶点时，三角形也存在。

The technique used to compute the number of triangles is as follows. 

用于计算三角形数量的技术如下。

Each vertex shares its neighbor list with each of its neighbors. 

每个顶点与其每个顶点共享其相邻列表。

Each vertex then computes the intersection between its neighbor list and the neighbor list(s) it receives. 

然后每个顶点计算其邻居列表与其接收到的邻居列表之间的交集。

For a given directed graph with no cycles, the size of the intersections gives the number of triangles in the graph.

对于没有周期的给定有向图，交点的大小给出了图中三角形的数量。

When the graph is undirected, then each vertex in a triangle contributes to the count, hence the size of the intersection is exactly 3 times the number of triangles. 

当图形是无向的，那么三角形中的每个顶点都有助于计数，因此交点的大小恰好是三角形数量的3倍。

The problem can be expressed mathematically as follows, where E uv denotes the presence of an (undirected) edge between vertex u and vertex v.

该问题可以用数学表达如下，其中E uv表示顶点u和顶点v之间存在（无向）边。

#### 3.5. Single Source Shortest Path (SSSP) 单源最短路径

This is another graph algorithm used to compute the shortest paths from a single source to all other vertices in a given weighted and directed graph. 

这是另一种图形算法，用于计算给定加权有向图中从单个源到所有其他顶点的最短路径。

The algorithm is used in many applications such as finding driving directions in maps or computing the min-delay path in telecommunication networks. 

该算法用于许多应用中，例如寻找地图中的驾驶方向或计算电信网络中的最小延迟路径。

Similar to BFS, the algorithm starts with a given vertex (called source) and iteratively explores all the vertices in the graph. 

与BFS类似，算法从给定的顶点（称为源）开始，并迭代地探索图中的所有顶点。

The idea is to assign a distance value to each vertex, which is the minimum edge weights needed to reach a particular vertex from the source. 

这个想法是为每个顶点分配一个距离值，这是从源头到达特定顶点所需的最小边权重。

At each iteration t, each vertex performs the following

在每次迭代t时，每个顶点执行以下操作

Where w(u,v) represents the weight of the edge (u,v). 

w（u，v）代表边的权重（u，v）。

Initially the Distance for each vertex is set to infinity except the source with Distance value set to 0. 

除了距离值设置为0的源之外，最初每个顶点的距离都设置为无穷大。

We use a slight variation on the Bellman-Ford shortest path algorithm where we only update the distance of those vertices that are adjacent to those that changed their distance in the previous iteration.

我们在Bellman-Ford最短路径算法上略有变化，我们只更新那些在前一次迭代中改变其距离的那些顶点的距离。

We now discuss the implementation of GraphMat and its opti-mizations in the next section.

我们现在讨论GraphMat的实现及其优化在下一节中介绍。

## 4. GRAPHMAT 图矩阵

## 5. RESULTS 实验结果

## 6. CONCLUSION AND FUTURE WORK 结论与后续工作