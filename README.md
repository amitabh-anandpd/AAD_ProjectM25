# Max-Flow Algorithms Empirical Comparison

This project provides implementations and empirical comparisons of several classic maximum flow algorithms:

- **Edmonds-Karp (Ford-Fulkerson with BFS)**
- **Ford-Fulkerson (with DFS and BFS variants)**
- **Dinic's Algorithm**
- **Push-Relabel**
- **Boykov-Kolmogorov (BK)**

## Project Structure

```
algorithms/
  fordfulkerson/   # Ford-Fulkerson (DFS & BFS/Edmonds-Karp)
  dinics/          # Dinic's Algorithm
  push_relabel/    # Push-Relabel Algorithm
Boykov-Kolmogorov/ # BK Algorithm (image & grid)
main.ipynb         # Main notebook for empirical comparison
```

## Algorithms

### Ford-Fulkerson (DFS & BFS/Edmonds-Karp)
- Located in `algorithms/fordfulkerson/code/`
- Supports both DFS (classic Ford-Fulkerson) and BFS (Edmonds-Karp) variants.
- Input: Graph files in simple edge-list format.

### Dinic's Algorithm
- Located in `algorithms/dinics/code/`
- Efficient for networks with many edges.

### Push-Relabel
- Located in `algorithms/push_relabel/code/`
- Uses the preflow-push method with gap relabeling heuristic.

### Boykov-Kolmogorov (BK)
- Located in `Boykov-Kolmogorov/`
- Designed for computer vision/image segmentation, but also supports synthetic grid graphs.

## Usage

- Use `main.ipynb` to run all algorithms on a common set of graphs and compare their empirical performance (runtime vs. edges/vertices).
- Each algorithm directory contains its own code, sample graphs, and results folders.
- BK algorithm can be run on both images and synthetic grid graphs; see its directory for details.

## Results & Visualization

- The notebook generates plots comparing runtime as a function of the number of edges and vertices for all algorithms.
- Results are saved in CSV files and visualized using matplotlib/seaborn.

## Requirements

- Python 3.7+
- See each algorithm's `requirements.txt` for dependencies (typically numpy, pandas, matplotlib, seaborn, tqdm).

## How to Run

1. Install dependencies:
   ```bash
   pip install -r algorithms/dinics/requirements.txt
   pip install -r algorithms/push_relabel/requirements.txt
   pip install -r algorithms/fordfulkerson/requirements.txt
   pip install -r Boykov-Kolmogorov/requirements.txt
   ```
2. Open and run `main.ipynb` for a full comparison and visualization.

## Notes

- Edmonds-Karp is implemented as the BFS variant of Ford-Fulkerson.
- All algorithms use a common graph file format for input (see code for details).
- For BK, both image and grid graph experiments are supported.

---

*For questions or contributions, please open an issue or pull request.*
