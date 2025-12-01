# Boykov-Kolmogorov Algorithm

By Yuri Boykov and Vladimir Kolmogorov in their famous paper “An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision”.

## 1. What is the Boykov-Kolmogorov Algorithm?

The Boykov-Kolmogorov Algorithm is an efficient way to compute max-flow for computer vision-related graphs. It is an augmenting paths-based algorithm which works by finding and pushing flow along shortest paths from a source to a sink in a graph until no more paths can be found.

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

### • `Growth` Stage

At this stage, the search trees (S and T) expand. The active nodes explore the adjecent non-saturated edges and acquire new children from a set of free nodes. The newly acquired nodes become active nodes of the tree. When all neighbours of an active node are explored, the active node becomes passive. The growth stage terminates when the two trees touch. The path fromed from source `s` to sink `t` fromed when the trees touch is passed to augmentation stage.

### • `Augmentation` Stage

This stage augments the path found in the growth stage. A flow is sent along the connecting path. Since the largest possible flow is sent, some edge(s) become saturated. The nodes which have saturated link to their parent node are referred to `orphans`. The orphan nodes are no longer available for passing more flow, hence splitting the trees into forest by making the orphan nodes as root nodes. 

### • `Adpotion` Stage

This stage fixes the forest to restore the single-tree structure of sets S and T with roots in the source and sink. This is done by finding new valid parent nodes for each orphan. The new parent should belong to the same set (S or T) as the orphan and should be connected through an unsaturated edge. If there is no qualifying parent, the orphan is removed from the set making it a free node and all its children become orphan nodes. This stage terminates when there are no orphan nodes remaining, restoring the S and T trees in the process. 

After the Adoption stage, the algorithm return to the growth stage. The algorithm terminates when search trees S and T cannot grow, i.e., there are no active nodes, and the trees are separated by saturated edges. This implies the maximum flow is achieved.

## 3. Pseudocode
```pseudocode
Input: Graph with nodes, edges, Source S, Sink T

Initialize:
    Label all nodes as FREE (unassigned)
    Label S as SOURCE-TREE
    Label T as SINK-TREE
    Add S and T to ACTIVE queue

    parent[node] = None for all nodes

Loop:
    1. GROW TREES
        While ACTIVE not empty:
            take node p from ACTIVE

            for each neighbor q of p:
                if residual capacity(p → q) > 0:
                    if q is FREE:
                        label q with same tree as p
                        parent[q] = p
                        add q to ACTIVE

                    else if q belongs to the opposite tree:
                        # Trees meet → we found a path from S to T
                        store meeting edge (p, q)
                        goto AUGMENT

    If no meeting edge found:
        break   # Trees can’t grow → done

    2. AUGMENT FLOW
        Build full path:
            S → ... → p — q ← ... ← T

        Find bottleneck (minimum residual capacity) along path

        Push flow = bottleneck through the path
        Some edges may become saturated → cause ORPHANS

        Add those ORPHANS to an ORPHAN queue

    3. ADOPTION (fix orphans)
        For each orphan o:
            Try to find a new parent in the same tree
                (a neighbor that can reach the root S or T)

            If found:
                parent[o] = that neighbor
            Else:
                mark o as FREE
                any children of o (in the tree) become new orphans

Repeat until ACTIVE empty and no meeting edge occurs

Output:
    All nodes reachable from S in the final residual graph = FOREGROUND
    All others = BACKGROUND
```

## 4. Time Complexity of Boykov-Kolmogorov Algorithm
### 4.1 Worst-Case Time Complexity
```math
T = O(mn^2|C|)
```
Where `n` is the number of nodes, `m` is the number of edges, and `|C|` is the cost of the minimum cut.

In an image - 
* $`n = H \times W + 2`$
* $`m \approx 4 \times H \times W`$
* |C| does not scale directly with H or W

Theoretically speaking, this is worse than the complexities of the standard algorithms.  However, experimental comparison shows that, on typical problem instances in vision, the Boykov-Kolmogorov algorithm significantly outperforms standard algorithms, and hence, the empirical complexity becomes linear.

### 4.2 Practical Complexity on Image Graphs

For typical computer vision graphs (sparse, grid-structured) the Boykov–Kolmogorov algorithm performs much better than the worst-case bound suggests. Empirically, the runtime is close to linear in the number of pixels:

```math
T_{\text{practical}} \approx O(m) \approx O(HW).
```
Boykov and Kolmogorov attribute this behaviour to several properties of vision graphs:

- augmenting paths are short,
- the S and T trees are reused and rarely break,
- the underlying graph is a regular 2D grid,
- effective path lengths grow at most \(O(1)\) or \(O(\log n)\).

These properties keep the number and cost of augmentations small, which is why BK significantly outperforms generic max-flow algorithms on such instances.

### 4.3 Space Complexity

The algorithm stores capacities, flows and adjacency lists for each edge and node. This leads to

```math
\text{Space} = O(m + n) \approx O(HW)
```

for grid-based image graphs.

### 4.4 Summary

- **Worst-case (general graphs):**  
  ```math
  T_{\text{worst}} = O(m\,n^{2}\,|C|).
  ```

- **For an `H × W` image (4-neighbour grid):**  
  ```math
  T_{\text{worst}} = O(H^{3}W^{3}|C|).
  ```

- **In practice on vision problems:**  
  ```math
  T_{\text{practical}} \approx O(HW) \quad \text{(essentially linear in the number of pixels)}.
  ```

- **Space complexity:**  
  ```math
  O(HW).
  ```

## 5. Benchmarking and Empirical Evaluation

Although the Boykov–Kolmogorov algorithm has a pessimistic theoretical worst-case bound, its primary significance lies in its empirical performance on vision-related graphs. To evaluate this behavior, we benchmark the algorithm on image-derived grid graphs of increasing size and measure its runtime as a function of graph size.

### 5.1 Benchmark Setup

To simulate typical computer vision graphs, we construct 4-connected grid graphs corresponding to images of size (H × W). Each pixel is represented as a node, with edges connecting neighboring pixels, as well as unary edges connecting pixels to the source and sink.

For each image size, we record:
- Number of vertices:
  ```math
  V = H \times W + 2
  ```
- Number of edges:
  ```math
  E \approx 4 \times H \times W
  ```
- Time required to compute the maximum flow using the Boykov–Kolmogorov algorithm

All experiments are conducted using the same implementation and environment to ensure fair comparison.

### 5.2 Metrics and Plots

The following metrics are used to analyze performance:

- **Runtime vs Image Size (H × W)**  
  Demonstrates how computation time scales with image resolution.

- **Runtime vs Number of Vertices (V)**  
  Relates performance directly to the theoretical problem size.

- **Runtime vs Number of Edges (E)**  
  Highlights the dependence of BK on graph connectivity.

Optionally, a log–log plot of runtime versus number of pixels may be used to estimate the empirical complexity exponent.

### 5.3 Empirical Observations

The benchmarking results show that:

- Runtime increases approximately linearly with the number of pixels.
- Doubling the image resolution results in roughly a proportional increase in computation time.
- No superlinear behavior predicted by the worst-case bound is observed.

These results confirm the claims made by Boykov and Kolmogorov that, despite unfavorable theoretical bounds, the algorithm performs efficiently on sparse, grid-structured graphs common in computer vision.

### 5.4 Discussion

The near-linear behavior observed in the benchmarks can be attributed to:
- Short augmenting paths in grid graphs,
- Limited orphan propagation,
- Reuse of search trees across augmentations,
- Regular and sparse graph structure.

This empirical efficiency explains why the Boykov–Kolmogorov algorithm remains a standard choice for energy minimization and segmentation tasks in computer vision.

## 6. Why Max-Flow Equals Min-Cut in the Boykov–Kolmogorov Algorithm

The Boykov–Kolmogorov (BK) algorithm is an augmenting-path-based **maximum flow** algorithm.  
Therefore, the classical **Max-Flow Min-Cut Theorem** applies:

```math
\text{Maximum flow value} = \text{Minimum s–t cut capacity}
```

The BK algorithm guarantees this equality through its structure and termination condition. This section explains conceptually and operationally why the result holds.

### 6.1 Flow Validity

At all times during execution, BK maintains a valid flow:

* Flow pushed only along edges with positive residual capacity

* Capacity constraints are never violated

* Flow conservation holds at all intermediate nodes

Thus, every augmentation produces another feasible flow.

### 6.2 Termination Condition and Residual Graph

The algorithm maintains two search trees:

* S-tree rooted at the source

* T-tree rooted at the sink

The algorithm terminates when neither tree can grow further, i.e., when:

$`There exists no path from source to sink in the residual graph`$

At termination, define:

* `S`: all nodes reachable from the source in the final residual graph

* `T = V \ S`

This `(S, T)` partition defines an s–t cut.

### 6.3 Why the Cut is Minimum

At termination:

1. All edges from S to T are saturated
  If an edge from S to T had remaining capacity, the T endpoint would be reachable from S, contradicting the definition of S.

2. All edges from T to S carry zero flow
Otherwise, there would exist a reverse residual edge allowing reachability.

3. Flow value equals cut capacity

```math
|f| = \sum_{u \in S,\, v \in T} c(u, v) = c(S, T)
```