# Boykov-Kolmogorov Algorithm

## 1. What is the Boykov-Kolmogorov Algorithm?

The Boykov-Kolmogorov Algorithm is an efficient way to compute max-flow for computer vision-related graphs. It is an augmenting paths-based algorithm which works by finding and pushing flow aloong shortest paths from a source to a sink in a graph until no more paths can be found. It was introduced by Yuri Boykov and Vladimir Kolmogorov in their famous paper “An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision”.

Similar to Dinic, it builds search trees. The difference is that BK algorithm builds two trees, one from the source and one from the sink. The two trees grow until they touch, giving a path from source to sink. Then it pushes flow along that path and updates the trees. It reuses the trees and never start building them from the scratch.

## 2. Algorithm's Overview

The algorithm maintains - 

1. `S tree`: Search tree with root at the source `s`.
2. `T tree`: Search tree with root at the sink `t`.
3. `Active nodes`: Outer border in each tree.
4. `Passive nodes`: Internal nodes of the trees.
5. `Orphan nodes`: Nodes whose edge linking to their parent becomes saturated, disconnecting them from their tree.

The algorithm iteratively repeats the following three stages - 

1. `Growth`: Expand the search trees simulteneously until they touch.
2. `Augmentation`: The path on which the trees touch is augmented and the trees are broken into forest.
3. `Adpotion`: Trees are restored by removing or reattaching the orphan nodes.

