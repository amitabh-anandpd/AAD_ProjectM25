# Push-Relabel Algorithm: Experimental Analysis Report

## Executive Summary

This report presents empirical analysis of the Push-Relabel algorithm
for computing maximum flow in directed graphs. The Push-Relabel algorithm
is fundamentally different from augmenting path methods (like Ford-Fulkerson
and Dinic's), as it maintains a preflow and works locally on vertices
rather than finding paths from source to sink.

## Dataset Overview

**Total Experiments:** 9

**Graph Families:** 6

| Family | Count |
|--------|-------|
| bidirectional | 1 |
| crosslinked | 1 |
| dense | 1 |
| layered | 1 |
| sample | 3 |
| sparse | 1 |

**Graph Sizes:** 6 - 10 vertices

## Performance Summary

### By Graph Family

| family        |   ('total_time', 'mean') |   ('total_time', 'std') |   ('total_time', 'min') |   ('total_time', 'max') |   ('num_pushes', 'mean') |   ('num_relabels', 'mean') |   ('num_operations', 'mean') |   ('max_flow', 'mean') |
|:--------------|-------------------------:|------------------------:|------------------------:|------------------------:|-------------------------:|---------------------------:|-----------------------------:|-----------------------:|
| bidirectional |                  2.03844 |               nan       |                 2.03844 |                 2.03844 |                        5 |                     5      |                            4 |                11      |
| crosslinked   |                  2.0003  |               nan       |                 2.0003  |                 2.0003  |                        7 |                     5      |                            4 |                15      |
| dense         |                  3.47761 |               nan       |                 3.47761 |                 3.47761 |                       11 |                     7      |                            7 |                18      |
| layered       |                  1.9288  |               nan       |                 1.9288  |                 1.9288  |                        6 |                     5      |                            4 |                14      |
| sample        |                  9.40555 |                 2.84358 |                 6.92545 |                12.5091  |                       26 |                    15.3333 |                           19 |                15.3333 |
| sparse        |                  2.94909 |               nan       |                 2.94909 |                 2.94909 |                       10 |                     7      |                            7 |                 4      |

## Algorithmic Insights

### Push-Relabel Characteristics

The Push-Relabel algorithm differs from path-based algorithms in several ways:

1. **Local Operations**: Works on individual vertices rather than global paths
2. **Preflow Maintenance**: Allows excess flow at intermediate vertices
3. **Height Function**: Uses distance labels to guide flow efficiently
4. **No Path Finding**: Doesn't explicitly find augmenting paths

**Average Push/Relabel Ratio:** 1.56

This ratio indicates how many push operations occur per relabel operation.
Higher ratios suggest that the height function effectively guides flow
without frequent relabeling.

## Scaling Analysis

### Empirical Scaling Laws

Fitted power-law relationships:

#### sample

- **Time vs Vertices (n):** `time ≈ 2.5440e+00 × n^0.62`
  - R² = 0.286

- **Time vs Edges (m):** `time ≈ 2.2881e+00 × m^0.53`
  - R² = 0.276

### Family-Specific Observations

#### bidirectional

- Average push operations: 5
- Average relabel operations: 5
- Average total operations: 4
- Average runtime: 2.038437s

#### crosslinked

- Average push operations: 7
- Average relabel operations: 5
- Average total operations: 4
- Average runtime: 2.000305s

#### dense

- Average push operations: 11
- Average relabel operations: 7
- Average total operations: 7
- Average runtime: 3.477609s

#### layered

- Average push operations: 6
- Average relabel operations: 5
- Average total operations: 4
- Average runtime: 1.928800s

#### sample

- Average push operations: 26
- Average relabel operations: 15
- Average total operations: 19
- Average runtime: 9.405545s

#### sparse

- Average push operations: 10
- Average relabel operations: 7
- Average total operations: 7
- Average runtime: 2.949091s

## Max-Flow Min-Cut Theorem Verification

The Push-Relabel implementation includes minimum cut computation from the
residual graph. For each experiment, the algorithm verifies that:

```
Maximum Flow Value = Minimum Cut Capacity
```

This empirically validates the max-flow min-cut theorem for all tested graphs.

## Visualizations

The following plots are available in the `visuals/` directory:

1. **plot_time_vs_n.png** - Runtime scaling with graph size
2. **plot_operations_vs_n.png** - Total operations vs graph size
3. **plot_push_relabel_ratio.png** - Push/Relabel operation ratio analysis
4. **summary.png** - Comparative summary across all graphs

Individual graph visualizations show:
- Initial network state
- After preflow initialization
- Intermediate states during algorithm execution
- Final maximum flow configuration

## Methodology

### Algorithm Variant

This implementation uses the **FIFO (First-In-First-Out)** vertex selection
heuristic, which provides good practical performance. Other variants include:
- Highest-label selection
- Excess scaling

### Graph Families

1. **Layered Graphs** - Nodes arranged in levels, edges flow left→right
2. **Cross-Linked Graphs** - Layered with additional cross edges
3. **Dense Graphs** - High edge density, many parallel paths
4. **Sparse Graphs** - Tree-like structure, minimal branching
5. **Bidirectional Graphs** - Forward and reverse edges present

### Metrics Collected

- `total_time`: Total algorithm runtime
- `num_pushes`: Number of push operations performed
- `num_relabels`: Number of relabel operations performed
- `num_operations`: Total operations (pushes + relabels)
- `max_flow`: Maximum flow value computed

## Theoretical Complexity

### Time Complexity

- **Generic Push-Relabel:** O(V²E)
- **With FIFO selection:** O(V³)
- **With highest-label:** O(V²√E)

### Space Complexity

- **O(V + E)** for storing the graph and auxiliary data structures
- Height array: O(V)
- Excess array: O(V)
- Active vertices queue: O(V)

## Conclusions

1. **Local Processing**: The Push-Relabel algorithm's local processing
   approach provides an alternative paradigm to path-based methods.

2. **Operation Characteristics**: The push/relabel ratio varies with
   graph structure, indicating how effectively the height function
   guides flow distribution.

3. **Min-Cut Computation**: The algorithm successfully computes minimum
   cuts, empirically validating the max-flow min-cut theorem.

4. **Graph Structure Impact**: Different graph families show varying
   performance characteristics, with dense graphs generally requiring
   more operations than sparse graphs.

## References

- Goldberg, A. V., & Tarjan, R. E. (1988). A new approach to the
  maximum-flow problem. Journal of the ACM, 35(4), 921-940.

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009).
  Introduction to Algorithms (3rd ed.). MIT Press.
