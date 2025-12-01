"""
Metrics collector for Ford–Fulkerson that mirrors Dinic's `metrics.py` API.
"""
import csv
import os
from typing import Dict, List


class MetricsCollector:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.iteration_metrics: List[Dict] = []

    def clear(self) -> None:
        self.iteration_metrics = []

    def record_iteration(self, iteration: int, path: List[int], flow_added: int, total_flow: int, bfs_time: float = 0.0, dfs_time: float = 0.0) -> None:
        self.iteration_metrics.append({
            "iteration": iteration,
            "path": path,
            "flow_added": flow_added,
            "total_flow": total_flow,
            "bfs_time": bfs_time,
            "dfs_time": dfs_time,
        })

    def write_performance_summary(self, graph_folder: str, graph_name: str, family: str, n: int, m: int, max_flow: int, min_cut_value: int, min_cut_edges: List[tuple], total_time: float, bfs_time_total: float, dfs_time_total: float, num_iterations: int, num_augmenting_paths: int) -> None:
        summary_path = os.path.join(self.results_dir, "performance.csv")
        header = [
            "graph_folder", "family", "graph_name", "n", "m", "total_time", "algorithm_time", 
            "bfs_time_total", "dfs_time_total", "num_iterations", "num_augmenting_paths", 
            "max_flow", "min_cut_value", "min_cut_edges"
        ]
        file_exists = os.path.exists(summary_path)
        if not file_exists:
            with open(summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)

        # Support cut edges represented as (u, v) or (u, v, c)
        if min_cut_edges:
            entries = []
            for e in min_cut_edges:
                if len(e) >= 2:
                    u, v = e[0], e[1]
                    c = e[2] if len(e) > 2 else None
                    if c is not None:
                        entries.append(f"{u}->{v}({c})")
                    else:
                        entries.append(f"{u}->{v}")
            min_cut_str = ";".join(entries)
        else:
            min_cut_str = ""
        algorithm_time = bfs_time_total + dfs_time_total
        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([graph_folder, family, graph_name, n, m, f"{total_time:.6f}", f"{algorithm_time:.6f}", f"{bfs_time_total:.6f}", f"{dfs_time_total:.6f}", num_iterations, num_augmenting_paths, max_flow, min_cut_value, min_cut_str])

    def write_detailed_iterations(self, graph_name: str) -> None:
        detailed_path = os.path.join(self.results_dir, "performance_iterations.csv")
        file_exists = os.path.exists(detailed_path)
        with open(detailed_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["graph", "iteration", "flow_added", "total_flow", "path"])
            for metric in self.iteration_metrics:
                path_str = " -> ".join(map(str, metric["path"]))
                writer.writerow([graph_name, metric["iteration"], metric["flow_added"], metric["total_flow"], path_str])
