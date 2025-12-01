# Dinic's Algorithm - Complete Workflow Guide

## Overview

The codebase is structured to separate algorithm execution from visualization generation. This allows you to:
- Run algorithms on large graphs (1000+ vertices) without generating expensive visualizations
- Generate visualizations on-demand only when needed
- Store execution data for later visualization
- Generate random graphs for scaling experiments

## Complete Workflow

### Step 0: Generate Random Graphs (Optional)

For scaling experiments, generate random graphs with specific characteristics:

```bash
python3 code/graph_generator.py
```

**What happens:**
- Generates **Type V graphs** in `graphs_v/`:
  - 15 graphs with constant edges (99) and increasing vertices
  - Graph 1: 100 vertices, 99 edges
  - Graph 2: 105 vertices, 99 edges
  - Graph 3: 110 vertices, 99 edges
  - ... up to Graph 15: 170 vertices, 99 edges

- Generates **Type E graphs** in `graphs_e/`:
  - 15 graphs with constant edges (200) and increasing vertices
  - Graph 1: 25 vertices, 200 edges
  - Graph 2: 30 vertices, 200 edges
  - Graph 3: 35 vertices, 200 edges
  - ... up to Graph 15: 95 vertices, 200 edges

**Graph Properties:**
- Source vertex: 0
- Sink vertex: n-1 (where n is the number of vertices)
- Edge capacities: Random between 1 and 100
- Each graph has a guaranteed path from source to sink
- Graphs are reproducible (seeded random generation)

**Customizing Graph Generation:**
Edit the parameters at the top of `code/graph_generator.py` to customize:
- Number of graphs to generate
- Starting vertices and edges
- Vertex increments
- Edge capacity ranges

### Step 1: Run Algorithm (NO Visualization)

#### Single Graph
```bash
python3 code/runner.py --graph graphs/sample1.txt --source 0 --sink 5
```

**Options:**
- `--graph`: Path to graph file (required)
- `--source`: Source vertex index (required)
- `--sink`: Sink vertex index (required)
- `--family`: Graph family name (optional, auto-detected if not provided)

**Example:**
```bash
python3 code/runner.py --graph graphs_v/type_v_01.txt --source 0 --sink 99 --family type_v
```

#### Batch Run (Multiple Graphs)

Run for a specific batch folder:
```bash
python3 code/batch_run.py graphs      # Process only graphs/ folder
python3 code/batch_run.py graphs_v    # Process only graphs_v/ folder
python3 code/batch_run.py graphs_e    # Process only graphs_e/ folder
```



**What happens:**
1. **Discovery**: Finds specified batch folder(s): `graphs/`, `graphs_v/`, `graphs_e/`
2. **Processing**: For each batch folder:
   - Processes all `.txt` graph files
   - Auto-detects source/sink (defaults to 0 and n-1)
   - Auto-detects family from filename
   - Runs Dinic's algorithm
   - Saves metrics and execution data
3. **Organization**: Results are organized by folder:
   - `results/graphs/` - for graphs from `graphs/` folder
   - `results/graphs_v/` - for graphs from `graphs_v/` folder
   - `results/graphs_e/` - for graphs from `graphs_e/` folder

**Note:** Cleanup is done manually. Before running batch experiments, clean up previous results:
```bash
rm -rf results/ visuals/ plots/
```

**Output:**
- Algorithm runs and computes maximum flow
- Metrics are saved to `results/performance.csv`
- Execution data is saved to `results/<folder>/<graph_name>_execution_data.json`
- **NO visualizations are generated** (saves time for large graphs)

### Step 2: Generate Visualizations (On-Demand)

After running the algorithm, generate visuals when needed:

#### Generate visuals for a specific batch:
```bash
python3 code/visualizer_script.py graphs      # Visualize only graphs/ batch
python3 code/visualizer_script.py graphs_v    # Visualize only graphs_v/ batch
python3 code/visualizer_script.py graphs_e    # Visualize only graphs_e/ batch
```
```

**What happens:**
- Loads execution data from `results/<folder>/<graph_name>_execution_data.json`
- Generates all visualization files in `visuals/<folder>/<graph_name>/`:
  - `initial_graph.png` - Initial flow network
  - `iteration_1/` - First iteration visualizations:
    - `initial_residual.png` - Residual graph at start of iteration
    - `selected_augmented_path.png` - Augmenting path found
    - `final_residual.png` - Residual graph after augmenting
  - `iteration_2/`, `iteration_3/`, ... - Subsequent iterations
  - `final_flow_graph.png` - Final flow network with flow values
  - `iteration_log.csv` - Detailed iteration log

**Visualization Features:**
- Level-based layout (deterministic positioning)
- Source node: Green
- Sink node: Purple
- Edge labels: `flow/capacity`
- Saturated edges: Red
- Reverse edges: Orange (dashed)
- Augmented path: Highlighted in cyan
- High DPI (250) for quality

### Step 3: Generate Analysis Table

After running experiments, generate analysis for a specific batch:

```bash
python3 code/analyze.py graphs      # Analyze only graphs/ batch
python3 code/analyze.py graphs_v    # Analyze only graphs_v/ batch
python3 code/analyze.py graphs_e    # Analyze only graphs_e/ batch
```

```

**What happens:**
- Loads performance data from `results/performance.csv` (filtered by batch if specified)
- Generates `results/<batch_folder>/summary_table.csv` (or `results/summary_table.csv` for all)
- Prints formatted table to console

**Table includes:**
- Graph name and family
- Vertices (n) and edges (m)
- Total time, algorithm time, BFS time, DFS time
- Number of iterations and augmenting paths
- Maximum flow and minimum cut value

### Step 4: Generate Scatter Plots

Generate performance analysis plots for a specific batch:

```bash
python3 code/plot.py graphs      # Plot only graphs/ batch
python3 code/plot.py graphs_v    # Plot only graphs_v/ batch
python3 code/plot.py graphs_e    # Plot only graphs_e/ batch
```


**What happens:**
- Loads performance data from `results/performance.csv` (filtered by batch if specified)
- Generates scatter plots in `plots/<batch_folder>/` (or `plots/` for all):
  - `scatter_time_vs_vertices.png` - Total time vs. number of vertices
  - `scatter_time_vs_edges.png` - Total time vs. number of edges
- Plots are color-coded by graph family

## Timing Clarification

### `total_time`
- **Wall-clock time** from start to finish
- Includes ALL overhead (function calls, data structures, I/O, etc.)
- Measured using `time.perf_counter()` around the entire algorithm execution
- This is the most accurate measure of actual runtime

### `algorithm_time`
- **Algorithm-only time** = `bfs_time_total + dfs_time_total`
- Tracks only the core algorithm operations (BFS and DFS phases)
- May be slightly less than `total_time` due to overhead

### `bfs_time_total`
- Total time spent in BFS level graph construction across all phases

### `dfs_time_total`
- Total time spent in DFS blocking flow computation across all phases

**Note:** For benchmarking and performance analysis, `total_time` is the most accurate measure of actual runtime.

## File Structure

### Project Directory Structure

```
dinics/
├── code/
│   ├── __init__.py
│   ├── dinics.py              # Pure algorithm implementation
│   ├── graph_loader.py        # Graph file loading
│   ├── graph_generator.py     # Random graph generation
│   ├── visualizer.py          # Visualization engine
│   ├── visualizer_script.py   # On-demand visualization script
│   ├── metrics.py             # Performance metrics collection
│   ├── data_store.py          # Execution data storage/loading
│   ├── runner.py              # Single graph runner
│   ├── batch_run.py           # Batch experiment runner
│   ├── analyze.py             # Analysis and summary table
│   ├── plot.py                # Scatter plot generation
├── graphs/                     # Original graph files
├── graphs_v/                   
├── graphs_e/                
├── results/
│   ├── performance.csv            # Summary metrics for all graphs
│   ├── performance_iterations.csv # Detailed per-iteration data
│   ├── summary_table.csv          # Formatted summary table
│   ├── output.txt                 # Last run's flow distribution
│   ├── run.log                    # Execution log
│   ├── graphs/                    # Execution data for graphs/ folder
│   ├── graphs_v/                  # Execution data for graphs_v/ folder
│   └── graphs_e/                  # Execution data for graphs_e/ folder
├── visuals/
│   ├── graphs/                    # Visualizations for graphs/ folder
│   ├── graphs_v/                  # Visualizations for graphs_v/ folder
│   └── graphs_e/                  # Visualizations for graphs_e/ folder
├── plots/                         # Scatter plots (organized by batch)
│   ├── graphs/                         # graphs/ batch plots
│   │   ├── scatter_time_vs_vertices.png
│   │   └── scatter_time_vs_edges.png
│   ├── graphs_v/                       # graphs_v/ batch plots
│   │   ├── scatter_time_vs_vertices.png
│   │   └── scatter_time_vs_edges.png
│   └── graphs_e/                       # graphs_e/ batch plots
│       ├── scatter_time_vs_vertices.png
│       └── scatter_time_vs_edges.png
├── README.md
├── WORKFLOW.md
└── requirements.txt
```

### Results Directory (`results/`)

**Root files:**
- `performance.csv` - Summary metrics for all graphs (one row per graph)
- `performance_iterations.csv` - Detailed per-iteration data (one row per augmenting path)
- `summary_table.csv` - Formatted summary table (generated by analyze.py)
- `output.txt` - Last run's flow distribution (detailed output)
- `run.log` - Execution log with timestamps

**Organized folders:**
- `results/graphs/` - Execution data JSON files and `summary_table.csv` for `graphs/` batch
- `results/graphs_v/` - Execution data JSON files and `summary_table.csv` for `graphs_v/` batch
- `results/graphs_e/` - Execution data JSON files and `summary_table.csv` for `graphs_e/` batch

### Execution Data Format

The `<graph_name>_execution_data.json` file contains:
```json
{
  "graph_name": "sample1",
  "graph_folder": "graphs",
  "num_vertices": 6,
  "source": 0,
  "sink": 5,
  "initial_edges": [[u, v, capacity], ...],
  "final_flow_dist": [[u, v, flow, capacity], ...],
  "path_history": [[iteration, path, flow_added], ...],
  "final_levels": [level0, level1, ...]
}
```

## Complete Example Workflow

### Example 1: Small Graph Analysis

1. **Run single graph:**
   ```bash
   python3 code/runner.py --graph graphs/sample1.txt --source 0 --sink 5
   ```

2. **Generate visualization:**
   ```bash
   python3 code/visualizer_script.py sample1
   ```

3. **View results:**
   - Check `results/output.txt` for flow distribution
   - Check `visuals/graphs/sample1/` for images

### Example 2: Batch Experiment with Random Graphs

1. **Generate random graphs:**
   ```bash
   python3 code/graph_generator.py
   ```
   This creates `graphs_v/` and `graphs_e/` folders with 15 graphs each.

2. **Run batch experiments for a specific batch:**
   ```bash
   python3 code/batch_run.py graphs_v
   ```
   This processes all graphs in `graphs_v/` folder only.

3. **Generate analysis for that batch:**
   ```bash
   python3 code/analyze.py graphs_v
   ```
   This creates `results/graphs_v/summary_table.csv` with metrics for that batch.

4. **Generate plots for that batch:**
   ```bash
   python3 code/plot.py graphs_v
   ```
   This creates scatter plots in `plots/graphs_v/` directory.

5. **Generate visualizations for that batch:**
   ```bash
   python3 code/visualizer_script.py graphs_v
   ```
   This generates all visuals for graphs in `graphs_v/` batch.

**Repeat steps 2-5 for other batches (`graphs`, `graphs_e`) as needed.**

### Example 3: Large Graph Workflow

For graphs with 1000+ vertices:

1. **Run algorithm** (fast, no visualization):
   ```bash
   python3 code/runner.py --graph huge_graph.txt --source 0 --sink 999
   ```

2. **Later, generate visuals for specific graph** (only if needed):
   ```bash
   python3 code/visualizer_script.py huge_graph
   ```

3. **Generate analysis** (aggregates all runs):
   ```bash
   python3 code/analyze.py
   ```

This approach saves significant time when running many large graphs, as visualization is the most expensive operation.

## Graph File Format

Graph files are simple text files with the following format:

```
num_vertices num_edges
u1 v1 capacity1
u2 v2 capacity2
...
```

**Example:**
```
6 8
0 1 10
0 2 5
1 3 8
1 4 2
2 3 3
2 4 7
3 5 12
4 5 4
```

- First line: Number of vertices and edges
- Subsequent lines: One edge per line (source, destination, capacity)
- Vertices are numbered from 0 to n-1
- Source and sink are typically 0 and n-1, but can be any vertices

## Benefits

- ✅ **Fast batch runs** - No visualization overhead during algorithm execution
- ✅ **On-demand visuals** - Generate only when needed
- ✅ **Scalable** - Works with graphs of any size (tested with 1000+ vertices)
- ✅ **Reproducible** - Execution data stored for later visualization
- ✅ **Flexible** - Can regenerate visuals with different settings later
- ✅ **Organized** - Results and visuals organized by graph folder
- ✅ **Random graph generation** - Easy generation of test graphs for scaling experiments
- ✅ **Comprehensive metrics** - Detailed timing and performance tracking

## Command Summary

| Command | Purpose |
|---------|---------|
| `python3 code/graph_generator.py` | Generate random graphs for experiments |
| `python3 code/runner.py --graph <file> --source <s> --sink <t>` | Run algorithm on single graph |
| `python3 code/batch_run.py [batch_folder]` | Run algorithm on graphs in specified batch folder (or all if not specified) |
| `python3 code/visualizer_script.py [batch_folder]` | Generate visualizations for specified batch (or all if not specified) |
| `python3 code/analyze.py [batch_folder]` | Generate summary table for specified batch (or all if not specified) |
| `python3 code/plot.py [batch_folder]` | Generate scatter plots for specified batch (or all if not specified) |

**Batch folders:** `graphs`, `graphs_v`, `graphs_e`

## Notes

- **Cleanup**: Cleanup is done manually using `rm -rf results/ visuals/ plots/`. This gives you full control over what to keep or remove.
- **Batch Folders**: `batch_run.py` processes only the main batch folders: `graphs/`, `graphs_v/`, and `graphs_e/`. Other folders are ignored.
- **Source/Sink Detection**: `batch_run.py` auto-detects source (0) and sink (n-1) for graphs not in the config. You can override by modifying `batch_run.py`.
- **Family Detection**: Graph families are auto-detected from filenames. Unknown graphs get empty family string.
- **Graph Generation**: Edit parameters at the top of `code/graph_generator.py` to customize graph generation (number of graphs, sizes, increments, etc.).
- **Performance**: For large graphs, visualization can take significant time. The separation of execution and visualization allows you to run experiments quickly and visualize selectively.
- **Plot Organization**: Plots can be organized by batch/folder manually. The `plot.py` script generates plots in the `plots/` directory.
`