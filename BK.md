# Boykov-Kolmogorov Algorithm

## 1. What is the Boykov-Kolmogorov Algorithm?

The Boykov-Kolmogorov Algorithm is an efficient way to compute max-flow for computer vision-related graphs. It is an augmenting paths-based algorithm which works by finding and pushing flow aloong shortest paths from a source to a sink in a graph until no more paths can be found. It was introduced by Yuri Boykov and Vladimir Kolmogorov in their famous paper “An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision”.

Similar to Dinic, it builds search trees. The difference is that BK algorithm builds two trees, one from the source and one from the sink. The two trees grow until they touch, giving a path from source to sink. Then it pushes flow along that path and updates the trees. It reuses the trees and never start building them from the scratch.

## 2. Algorithm's Overview

