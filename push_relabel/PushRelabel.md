# Push-Relabel Algorithm (Sami)

Implementation of the Push-Relabel algorithm for computing maximum flow in directed graphs.

## Overview

The Push-Relabel algorithm represents a fundamentally different approach to the maximum flow problem compared to augmenting path methods. Instead of finding complete paths from source to sink, it works locally on vertices, maintaining a "preflow" that may have excess at intermediate nodes.

## Why Push-Relabel?

### Theoretical Advantages

1. **Better Worst-Case Complexity**: O(V³) with FIFO selection vs O(V²E) for basic augmenting path methods
2. **Parallelizable**: Local operations can be performed independently
3. **No Path Finding**: Avoids expensive path searches through the entire graph

### Key Concepts

#### Preflow
Unlike a valid flow, a preflow allows vertices (except source and sink) to have excess flow:
- **Excess**: The amount by which flow into a vertex exceeds flow out
- **Valid when**: excess[v] ≥ 0 for all vertices v

#### Height Function
Each vertex has a "height" label that guides flow:
- Source starts at height n (number of vertices)
- Flow is "pushed" from higher to lower vertices
- Heights are adjusted via "relabel" operations when needed

#### Two Core Operations

1. **Push**: Send excess flow from a vertex to a lower neighbor
   - Can only push to a vertex with height exactly 1 less
   - Push amount = min(excess, residual capacity)

2. **Relabel**: Increase a vertex's height when it can't push anywhere
   - New height = 1 + min(height of pushable neighbors)

### The Algorithm

```
1. Initialize preflow:
   - Set source height to n
   - Saturate all edges from source
   - Give excess to source's neighbors

2. While there exist active vertices (with excess):
   - Select an active vertex u
   - Discharge u:
     * Try to push excess to neighbors
     * If no push possible, relabel u
     * Repeat until u has no excess

3. Return flow value = Σ flow(source, v)
```

### FIFO Selection Heuristic

This implementation uses **FIFO (First-In-First-Out)** for selecting active vertices:
- Maintains a queue of active vertices
- Process vertices in the order they became active
- Achieves O(V³) time complexity
- Good practical performance

### Min-Cut Computation

After computing max flow, the minimum cut is found by:
1. BFS from source in residual graph
2. Source side S = all reachable vertices
3. Sink side T = all unreachable vertices
4. Cut edges = edges from S to T in original graph

This empirically validates the **Max-Flow Min-Cut Theorem**:
```
Maximum Flow Value = Minimum Cut Capacity
```

## Project Structure

```
push_relabel/
├── code/
│   ├── push_relabel.py      # Core algorithm implementation
│   ├── runner.py             # CLI runner for single graphs
│   ├── batch_run.py          # Batch experiment runner
│   ├── analyze.py            # Analysis and plotting tools
│   └── generate_report.py    # Report generation
├── graphs/
│   ├── sample1.txt           # 6-node sample graph
│   ├── sample2.txt           # 8-node sample graph
│   ├── sample3.txt           # 10-node sample graph
│   ├── family_A_layered.txt
│   ├── family_B_crosslinked.txt
│   ├── family_C_dense.txt
│   ├── family_D_sparse.txt
│   └── family_E_bidirectional.txt
├── results/
│   ├── performance.csv       # Summary metrics
│   ├── output.txt            # Flow distribution and min-cut
│   └── run.log               # Execution log
├── visuals/
│   ├── [graph_name]/
│   │   ├── initial_graph.png
│   │   ├── after_preflow.png
│   │   ├── operation_*.png
│   │   └── final_flow_graph.png
│   └── [plots from analyze.py]
└── requirements.txt
```

## Installation

```bash
# Install dependencies
pip install --user networkx matplotlib pandas numpy scipy tabulate

# Or use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Single Graph Execution

```bash
python3 code/runner.py --graph graphs/sample1.txt --source 0 --sink 5
```

### Batch Experiments

Run all sample and family graphs:

```bash
python3 code/batch_run.py
```

### Generate Analysis

After running experiments, generate plots and analysis:

```bash
python3 code/analyze.py
```

This creates:
- `plot_time_vs_n.png` - Runtime scaling with graph size
- `plot_operations_vs_n.png` - Total operations vs size
- `plot_push_relabel_ratio.png` - Push/Relabel ratio analysis
- `summary.png` - Comparative summary

### Generate Report

Create comprehensive experimental report:

```bash
python3 code/generate_report.py
```

Generates `EXPERIMENTAL_REPORT.md` with:
- Performance summaries
- Scaling law fits
- Family-specific observations
- Algorithmic insights
- Min-cut validation

## Graph File Format

```
<vertices> <edges>
<u> <v> <capacity>
<u> <v> <capacity>
...
```

Example:
```
6 10
0 1 10
0 2 8
1 2 5
...
```

## Visualization Features

### Node Coloring
- **Light Green** - Source node
- **Medium Purple** - Sink node
- **Light Coral** - Active vertices (excess > 0)
- **Gold** - Currently processing vertex
- **White** - Inactive vertices

### Node Labels
Each node displays:
- Vertex ID
- Height (h=...)
- Excess flow (e=...)

### Edge Coloring
- **Gray** - Unused edges (no flow)
- **Royal Blue** - Flowing edges (0 < flow < capacity)
- **Firebrick** - Saturated edges (flow = capacity)

## Performance Metrics

The algorithm tracks:
- Total runtime
- Number of push operations
- Number of relabel operations
- Total operations (push + relabel)
- Maximum flow value
- Minimum cut edges and capacity

## Theoretical Complexity

### Time Complexity
- **Generic Push-Relabel**: O(V²E)
- **FIFO selection**: O(V³)
- **Highest-label selection**: O(V²√E)

### Space Complexity
- **O(V + E)** for graph storage
- **O(V)** for height, excess, and queue

## Comparison with Other Algorithms

| Algorithm | Time Complexity | Space | Approach |
|-----------|----------------|-------|----------|
| Ford-Fulkerson | O(E·f) | O(V+E) | Augmenting paths |
| Dinic's | O(V²E) | O(V+E) | Level graph + blocking flow |
| Push-Relabel | O(V³) | O(V+E) | Local vertex operations |

Where:
- V = vertices
- E = edges  
- f = max flow value

## References

1. Goldberg, A. V., & Tarjan, R. E. (1988). A new approach to the maximum-flow problem. *Journal of the ACM*, 35(4), 921-940.

2. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.

3. Ahuja, R. K., Magnanti, T. L., & Orlin, J. B. (1993). *Network Flows: Theory, Algorithms, and Applications*. Prentice Hall.
