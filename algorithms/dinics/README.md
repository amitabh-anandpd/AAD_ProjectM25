# Dinic’s Algorithm (Arun)
Implementation folder for Dinic’s Algorithm.

Why I'm Focusing on Dinic's Algorithm

When I was diving into network flow, I wanted to find the fastest way to calculate the maximum flow in a graph. Dinic's algorithm is a game-changer.

It's an extremely fast, strongly polynomial algorithm. This is a big deal because it means its runtime doesn't depend on the (potentially massive) capacity values on the edges.

What's really remarkable is its performance on bipartite graphs. It can solve the unweighted bipartite matching problem in O(√V⋅E) time, which is fast enough to handle ridiculously large graphs. If you're doing competitive programming, this is almost always the algorithm you'll want to use for max-flow.

It was revolutionary because it introduced several new concepts all at once:

Building a level graph.

Finding a blocking flow.

Combining different graph traversals (BFS and DFS) in a clever way.

☕ The Main Idea: The "Coffee Shop" Analogy

Before diving into the technical steps, here's an analogy I like.

Imagine you're at a starting point (the source, S) and you want to get to a coffee shop (the sink, T). You don't know the exact path, but you know the coffee shop is generally east of you.

If you want to get there, would you start by walking south? Or northwest? Probably not. The only sensible directions are east, northeast, and southeast. You'd use a heuristic: only move in directions that make positive progress toward your goal.

This is the central idea of Dinic's. We don't want to waste time exploring paths that take us further away from the sink. We need a way to guide our search.

📈 The Level Graph (Our Guiding Heuristic)

Dinic's algorithm creates this "guiding" heuristic by building a level graph.

Here’s how it works:

We run a Breadth-First Search (BFS) starting from the source (S) on the current residual graph.

The "level" of any node is its shortest path distance (in number of edges) from S. So, S is at level 0, its direct neighbors are at level 1, their neighbors are at level 2, and so on.

The level graph only includes edges (u, v) that go from a node u at level L to a node v at level L+1.

This is our "coffee shop" rule! It instantly prunes all useless edges:

Backwards edges (going from L+1 to L) are ignored.

Sideways edges (going from L to L) are ignored.

We only ever move "forward" toward the sink, guaranteeing we're making progress. We also, of course, only consider edges that have a remaining capacity greater than zero.

🛠️ The Algorithm: Steps and Blocking Flows

The algorithm works in phases. In each phase, we find a "blocking flow" and add it to our total. We repeat this until no more flow can be sent.

Here are the steps:

Step 1: Build the Level Graph

Run a BFS from the source (S) on the current residual graph to find the level of every node.

Step 2: Check if Sink is Reachable

After the BFS, if the sink (T) was not reached, it means there is no path left from S to T.

We're done! The algorithm terminates, and we return the total max flow we've found so far.

Step 3: Find a Blocking Flow

If the sink was reached, we now use our new level graph.

We find augmenting paths from S to T by running one or more Depth-First Searches (DFS).

Crucially: The DFS is only allowed to use edges in the level graph (i.e., edges from level L to L+1).

For each path we find:

Calculate its bottleneck capacity (the smallest remaining capacity on the path).

Add this bottleneck value to our total max flow.

Update the residual capacities along the path (decreasing forward capacity, increasing backward capacity).

Step 4: Repeat until Blocked

We keep running this DFS (Step 3) and pushing flow until we cannot find any more S-T paths in the current level graph.

This state is called a blocking flow. It means we've "saturated" the level graph—at least one edge on every S-T path in that specific level graph is now full.

Step 5: Repeat the Whole Process

Once a blocking flow is reached, we discard the old level graph.

We go back to Step 1 and build a brand new level graph based on the current residual capacities.

We repeat this entire process (Build Level Graph -> Find Blocking Flow) until the BFS in Step 2 fails to reach the sink.

⚡ A Critical Optimization: Pruning Dead Ends

There's one last trick that makes this algorithm incredibly fast.

During the DFS phase (Step 3), what happens if we explore a path that leads to a "dead end"? (A node from which we can't reach the sink, because all its forward edges are saturated).

It would be very inefficient to re-explore this same dead-end path multiple times during the same blocking flow phase.

The solution is dead-end pruning. As our DFS backtracks from a node u because it's a dead end, we can effectively "prune" u. We mark it (or use a pointer system) so we don't bother visiting it again during this blocking flow phase. This ensures we only explore each "bad" path once, which simplifies the algorithm and speeds it up dramatically.

📜 Summary

So, to recap, Dinic's algorithm is powerful because it cleverly combines:

BFS to build a level graph, which ensures we only make progress toward the sink.

DFS (with dead-end pruning) to find a blocking flow by efficiently pushing as much flow as possible through that level graph.

It repeats this process in phases, rebuilding the level graph each time until the max flow is found.


## Features

- ✅ **Complete Dinic's Algorithm Implementation** - Level-based BFS + blocking flow DFS
- ✅ **Rich Visualizations** - Per-iteration graphs showing residual networks and augmenting paths
- ✅ **Level-Based Layout** - Nodes positioned by BFS level for intuitive understanding
- ✅ **Performance Metrics** - Detailed timing and iteration tracking
- ✅ **Minimum Cut Extraction** - Automatically computes min-cut after max-flow
- ✅ **Modular Architecture** - Clean separation of algorithm, visualization, and metrics

## Project Structure

```
dinics/
├── code/
│   ├── __init__.py
│   ├── dinics.py          # Pure algorithm implementation
│   ├── graph_loader.py    # Graph file loading
│   ├── visualizer.py      # Visualization engine
│   ├── metrics.py         # Performance metrics collection
│   ├── runner.py          # Single graph runner
│   ├── batch_run.py       # Batch experiment runner
│   └── analyze.py         # Analysis and summary table
├── graphs/
│   ├── sample1.txt         # 6-node sample graph
│   ├── sample2.txt         # 8-node sample graph
│   ├── sample3.txt         # 10-node sample graph
│   ├── layered_1.txt       # Layered graph (6 nodes)
│   ├── layered_2.txt       # Layered graph (8 nodes)
│   ├── layered_3.txt       # Layered graph (10 nodes)
│   ├── crosslinked_1.txt   # Crosslinked graph (7 nodes)
│   ├── crosslinked_2.txt   # Crosslinked graph (8 nodes)
│   ├── crosslinked_3.txt   # Crosslinked graph (10 nodes)
│   ├── dense_1.txt         # Dense graph (6 nodes)
│   ├── dense_2.txt         # Dense graph (8 nodes)
│   ├── dense_3.txt         # Dense graph (10 nodes)
│   ├── sparse_1.txt        # Sparse graph (6 nodes)
│   ├── sparse_2.txt        # Sparse graph (8 nodes)
│   ├── sparse_3.txt        # Sparse graph (10 nodes)
│   ├── bidirectional_1.txt # Bidirectional graph (6 nodes)
│   ├── bidirectional_2.txt # Bidirectional graph (8 nodes)
│   └── bidirectional_3.txt # Bidirectional graph (10 nodes)
├── results/
│   ├── performance.csv            # Summary metrics
│   ├── performance_iterations.csv # Per-iteration details
│   ├── summary_table.csv          # Formatted summary table
│   ├── output.txt                 # Flow distribution
│   └── run.log                    # Execution log
├── visuals/
│   ├── sample1/
│   │   ├── initial_graph.png
│   │   ├── final_flow_graph.png
│   │   ├── iteration_1/
│   │   │   ├── initial_residual.png
│   │   │   ├── selected_augmented_path.png
│   │   │   └── final_residual.png
│   │   └── iteration_log.csv
│   └── [similar folders for each graph]
└── requirements.txt
```

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

```bash
# Install dependencies using pip
pip install networkx matplotlib pandas numpy scipy

# Or use a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Single Graph Execution

Run Dinic's algorithm on a single graph file:

```bash
python3 code/runner.py --graph graphs/sample1.txt --source 0 --sink 5
```

**Options:**
- `--graph`: Path to graph file (e.g., `graphs/sample1.txt`)
- `--source`: Source vertex index (default: 0)
- `--sink`: Sink vertex index (required)
- `--family`: Graph family name (optional, auto-detected if not provided)

**Example:**
```bash
python3 code/runner.py --graph graphs/layered_1.txt --source 0 --sink 5
```

### Batch Experiments

Run Dinic's algorithm on all graphs in the `graphs/` directory:

```bash
python3 code/batch_run.py
```

This will:
- Auto-detect graph family from filename
- Process all `.txt` files in the `graphs/` directory
- Generate visualizations for each graph
- Collect performance metrics
- Save results to `results/performance.csv`

### Generate Analysis Table

After running experiments, generate a summary table:

```bash
python3 code/analyze.py
```

This creates:
- `results/summary_table.csv` - Summary table with all performance metrics
- Prints a formatted table to console

### Recommended Workflow

1. **Run experiments:**
   ```bash
   python3 code/batch_run.py
   ```

2. **Generate analysis:**
   ```bash
   python3 code/analyze.py
   ```

## Graph File Format

Graph files should follow this format:

```
<num_vertices> <num_edges>
<u> <v> <capacity>
<u> <v> <capacity>
...
```

**Example (`graphs/sample1.txt`):**
```
6 10
0 1 10
0 2 8
1 2 5
1 3 5
2 4 10
3 4 8
3 5 7
4 5 10
1 4 2
2 3 3
```

## Graph Families

The project includes the following graph families:

- **Sample Graphs** (`sample1.txt`, `sample2.txt`, `sample3.txt`) - Basic test cases
- **Layered Graphs** (`layered_1.txt`, `layered_2.txt`, `layered_3.txt`) - Clear level structure, great for Dinic's blocking flow
- **Crosslinked Graphs** (`crosslinked_1.txt`, `crosslinked_2.txt`, `crosslinked_3.txt`) - Layered with crossing edges to create extra paths
- **Dense Graphs** (`dense_1.txt`, `dense_2.txt`, `dense_3.txt`) - High connectivity, many edges
- **Sparse Graphs** (`sparse_1.txt`, `sparse_2.txt`, `sparse_3.txt`) - Tree-like structure, minimal edges
- **Bidirectional Graphs** (`bidirectional_1.txt`, `bidirectional_2.txt`, `bidirectional_3.txt`) - Forward and reverse edges for testing residual mechanics

## Visualization Features

### Color Coding
- **Green** - Source node
- **Purple** - Sink node
- **Light Blue** - Intermediate nodes
- **Cyan** - Augmenting path edges (highlighted)
- **Red** - Saturated edges (flow = capacity)
- **Royal Blue** - Flowing edges (flow > 0)
- **Gray** - Unused edges
- **Orange (dashed)** - Reverse residual edges

### Visual Elements
- **Edge Labels**: Format `flow/capacity` on each edge
- **Level Labels**: Each node displays its BFS level (L0, L1, L2...) or ∞ if unreachable
- **Curved Edges**: Prevents overlapping for better readability
- **High DPI**: 250 DPI for report-quality images
- **Residual Graphs**: Dashed lines for residual edges

## Performance Metrics

The algorithm tracks:
- `total_time` - Total algorithm runtime
- `bfs_time_total` - Cumulative BFS phase time
- `dfs_time_total` - Cumulative DFS traversal time
- `num_iterations` - Number of BFS phases
- `num_augmenting_paths` - Total augmenting paths found
- `max_flow` - Maximum flow value
- `min_cut_value` - Minimum cut value
- `min_cut_edges` - List of edges in the minimum cut

## Output Structure

### Results Directory (`results/`)
- `performance.csv` - Summary metrics for all graphs (family, graph_name, n, m, timings, flow, min_cut)
- `performance_iterations.csv` - Per-iteration details for each graph
- `summary_table.csv` - Formatted summary table (generated by analyze.py)
- `output.txt` - Flow distribution and min-cut for the last run
- `run.log` - Execution log

### Visuals Directory (`visuals/`)
For each graph, a folder is created with:
- `initial_graph.png` - Initial flow network
- `final_flow_graph.png` - Final flow network with max flow
- `iteration_1/`, `iteration_2/`, ... - Per-iteration visualizations:
  - `initial_residual.png` - Residual network before path
  - `selected_augmented_path.png` - Highlighted augmenting path
  - `final_residual.png` - Residual network after path
- `iteration_log.csv` - Detailed iteration metrics




