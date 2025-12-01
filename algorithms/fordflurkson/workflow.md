# Ford–Fulkerson - Complete Workflow Guide

This folder mirrors the Dinic workflow but for the Ford–Fulkerson implementations
(`classical` DFS-based and `bfs`/Edmonds–Karp). It provides graph generation,
batch runs, visualization, plotting, and analysis tools.

## Quick commands

- Generate graphs (use Dinic's generator in `algorithms/dinics`)
  ```bash
  python3 ../dinics/code/graph_generator.py
  ```
- Sync generated graphs into Ford–Fulkerson folder (copies into `classical/` and `edmondson/`):
  ```bash
  bash ../scripts/script.sh
  ```
- Run single graph with chosen traversal method:
  ```bash
  python3 code/runner.py --graph <path> --source 0 --sink N-1 --method dfs
  python3 code/runner.py --graph <path> --source 0 --sink N-1 --method bfs
  ```
- Batch-run a folder of graphs (mirrors Dinic `batch_run` behavior):
  ```bash
  python3 code/batch_run.py <batch_folder>
  ```
- Generate visualizations from saved execution data:
  ```bash
  python3 code/visualizer_script.py <batch_folder>
  ```
- Generate analysis and plots:
  ```bash
  python3 code/analyze.py <batch_folder>
  python3 code/plot.py <batch_folder>
  ```

Notes:
- Execution data is stored under `algorithms/fordflurkson/results/<method>/...` and
  mirrored into `results/classical/` and `results/edmondson/` for convenience.
- Visualizations and plots are created from the saved execution JSONs and do not
  require re-running the algorithm.


# make script executable
chmod +x scripts/script.sh

# replace incorrect symlink with one that points to the absolute script path
rm -f scripts/sync_graphs_from_dinics.sh
ln -sf "$(pwd)/scripts/script.sh" scripts/sync_graphs_from_dinics.sh

# verify directory and run dry-run
ls -la scripts
DRY_RUN=1 ./scripts/script.sh