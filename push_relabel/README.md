# Push-Relabel Algorithm Implementation

This directory contains a complete implementation and analysis of the **Push-Relabel algorithm** for computing maximum flow in directed graphs.

## Project Overview

This project is part of a larger study on Maximum Flow and Minimum Cut algorithms, comparing different approaches:
1. Ford-Fulkerson (baseline augmenting path method)
2. Dinic's algorithm (implemented by Arun)
3. **Push-Relabel algorithm (this implementation)**

## What's Implemented

### Core Algorithm (✓ Complete)
- Push-Relabel with FIFO vertex selection
- Preflow initialization
- Push and relabel operations
- Efficient data structures (height, excess, active queue)

### Min-Cut Computation (✓ Complete)
- BFS-based min-cut extraction from residual graph
- Validation of max-flow min-cut theorem
- Cut edge and capacity reporting

### Visualizations (✓ Complete)
- Initial graph state
- After preflow initialization
- Intermediate algorithm states
- Final max flow configuration
- Node labels with height and excess
- Color-coded edges by flow status

### Performance Analysis (✓ Complete)
- Runtime measurements
- Operation counting (push/relabel)
- Scaling analysis across graph families
- Comparative metrics generation

### Experimental Framework (✓ Complete)
- Batch processing of multiple graphs
- CSV-based metrics collection
- Automated plot generation
- Comprehensive report generation

## Quick Start

### 1. Install Dependencies
```bash
cd algorithms/push_relabel
pip install -r requirements.txt
```

### 2. Run on a Single Graph
```bash
python3 code/runner.py --graph graphs/sample1.txt --source 0 --sink 5
```

### 3. Run Batch Experiments
```bash
python3 code/batch_run.py
```

### 4. Generate Analysis
```bash
python3 code/analyze.py
```

### 5. Generate Report
```bash
python3 code/generate_report.py
```

## Directory Structure

```
Sami_PushRelabel/
└── algorithms/
    └── push_relabel/
        ├── code/
        │   ├── __init__.py
        │   ├── push_relabel.py       # Core algorithm
        │   ├── runner.py              # Single graph runner
        │   ├── batch_run.py           # Batch experiments
        │   ├── analyze.py             # Analysis tools
        │   └── generate_report.py     # Report generator
        ├── graphs/
        │   ├── sample*.txt            # Sample graphs
        │   └── family_*.txt           # Graph families
        ├── results/
        │   ├── performance.csv        # Metrics
        │   └── output.txt             # Flow distribution
        ├── visuals/
        │   └── [generated plots]
        ├── PushRelabel.md             # Algorithm documentation
        ├── EXPERIMENTAL_REPORT.md     # Generated report
        └── requirements.txt
```

## Key Features

### Algorithm Features
- **FIFO selection heuristic** for O(V³) complexity
- **Efficient push operations** with residual capacity checks
- **Height-based flow guidance** for optimal performance
- **Min-cut computation** with theorem verification

### Analysis Features
- **Scaling law fits** (time vs n, time vs m)
- **Operation analysis** (push/relabel ratios)
- **Family-specific insights**
- **Visual performance comparisons**

### Visualization Features
- **Node coloring** by role (source/sink/active/inactive)
- **Height and excess labels** on each vertex
- **Edge coloring** by flow status
- **Periodic snapshots** during execution

## Graph Families

The implementation is tested on diverse graph families:

1. **Layered** - Nodes in levels, forward edges only
2. **Crosslinked** - Layered with additional cross-edges
3. **Dense** - High edge density, many parallel paths
4. **Sparse** - Tree-like, minimal branching
5. **Bidirectional** - Forward and reverse edges

## Output Files

### results/output.txt
```
Maximum Flow: <value>

Flow Distribution:
u → v : flow / capacity
...

Minimum Cut:
Source Side (S): [...]
Sink Side (T): [...]
Cut Edges: [...]
Cut Capacity: <value>
Max-Flow Min-Cut Theorem Verified: True

Runtime: <time>s
Number of Push Operations: <count>
Number of Relabel Operations: <count>
```

### results/performance.csv
```
graph_filename,family,n,m,seed,trial,max_flow,total_time,num_pushes,num_relabels,num_operations
...
```

## Theoretical Background

### Time Complexity
- Generic: O(V²E)
- FIFO: O(V³)
- Highest-label: O(V²√E)

### Space Complexity
- O(V + E) total
- Graph: O(V + E)
- Auxiliary structures: O(V)

### Key Properties
1. Maintains preflow (excess ≥ 0)
2. Height function guides flow
3. No explicit path finding
4. Local vertex operations
5. Parallelizable

## Comparison Goals

This implementation will be compared with:

- **Dinic's algorithm** (Arun's implementation)
- **Ford-Fulkerson** (Nikila's implementation)
- **Boykov-Kolmogorov** (Amitabh's implementation) 

### Comparison Metrics
- Runtime vs graph size
- Memory usage
- Number of operations
- Scalability with density
- Performance on different graph families

## Future Enhancements

1. **Larger graphs** - Scale up to 100+ vertices
2. **Random graph generation** - Configurable density and capacity
3. **Additional heuristics** - Highest-label, excess scaling
4. **Parallel implementation** - Exploit local operations
5. **Comparative analysis** - Side-by-side with Dinic's

## References

- Goldberg & Tarjan (1988) - Original Push-Relabel paper
- CLRS (2009) - Introduction to Algorithms
- Ahuja, Magnanti & Orlin (1993) - Network Flows