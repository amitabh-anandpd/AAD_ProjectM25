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
### Worst-Case Time Complexity
$$
T = O(mn^2|C|)
$$
Where `n` is the number of nodes, `m` is the number of edges, and `|C|` is the cost of the minimum cut.
In an image - 
* $`n = H * W + 2`$