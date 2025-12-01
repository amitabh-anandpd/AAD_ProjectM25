"""
Metrics collection and performance logging module.

This module handles collecting timing and performance statistics from
Dinic's algorithm runs and writing them to CSV files.
"""
import csv
import os
from typing import Dict, List, Optional


class MetricsCollector:
    """
    Collector for algorithm performance metrics.
    
    In this class, I track timing information (BFS, DFS, total runtime),
    algorithmic metrics (iterations, augmenting paths), and results
    (max flow, min cut).
    """
    
    def __init__(self, results_dir: str):
        """
        Initialize metrics collector.
        
        Args:
            results_dir: Directory where results CSV files will be written
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.iteration_metrics: List[Dict] = []
    
    def clear(self) -> None:
        """Clear all collected metrics for a new run."""
        self.iteration_metrics = []
    
    def record_iteration(self, iteration: int, path: List[int], flow_added: int, 
                        total_flow: int, bfs_time: float = 0.0, dfs_time: float = 0.0) -> None:
        """
        Record metrics for a single iteration (augmenting path).
        
        Args:
            iteration: Iteration number (1-indexed)
            path: List of nodes in the augmenting path
            flow_added: Flow added by this path
            total_flow: Total flow after this iteration
            bfs_time: BFS time for this phase (only recorded once per phase)
            dfs_time: DFS time for finding this path
        """
        self.iteration_metrics.append({
            "iteration": iteration,
            "path": path,
            "flow_added": flow_added,
            "total_flow": total_flow,
            "bfs_time": bfs_time,
            "dfs_time": dfs_time,
        })
    
    def write_iteration_log(self, graph_name: str, visuals_dir: str) -> None:
        """
        Write detailed iteration log to CSV file in the graph's visuals directory.
        
        Args:
            graph_name: Name of the graph (without extension)
            visuals_dir: Directory where visualizations are stored
        """
        log_dir = os.path.join(visuals_dir, graph_name)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "iteration_log.csv")
        
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iteration", "bfs_time", "dfs_time", "flow_added", 
                "total_flow_after", "num_augmenting_paths"
            ])
            for metric in self.iteration_metrics:
                path_str = " -> ".join(map(str, metric["path"]))
                writer.writerow([
                    metric["iteration"],
                    f"{metric.get('bfs_time', 0):.6f}",
                    f"{metric.get('dfs_time', 0):.6f}",
                    metric["flow_added"],
                    metric["total_flow"],
                    1,  # One path per iteration
                ])
    
    def write_performance_summary(
        self,
        graph_name: str,
        graph_folder: str,
        family: str,
        n: int,
        m: int,
        max_flow: int,
        min_cut_value: int,
        min_cut_edges: List[tuple],
        total_time: float,
        bfs_time_total: float,
        dfs_time_total: float,
        num_iterations: int,
        num_augmenting_paths: int,
    ) -> None:
        """
        Append performance summary to results/performance.csv.
        
        This block writes a single row with all performance metrics for one graph run.
        The CSV format is:
            family,graph_name,n,m,total_time,bfs_time_total,dfs_time_total,
            num_iterations,num_augmenting_paths,max_flow,min_cut_value,min_cut_edges
        """
        summary_path = os.path.join(self.results_dir, "performance.csv")
        
        # Format min cut edges as string representation
        min_cut_str = ";".join(f"{u}->{v}" for u, v in min_cut_edges) if min_cut_edges else ""
        
        header = [
            "graph_folder", "family", "graph_name", "n", "m", "total_time", "algorithm_time", 
            "bfs_time_total", "dfs_time_total", "num_iterations", "num_augmenting_paths", 
            "max_flow", "min_cut_value", "min_cut_edges"
        ]
        
        # Check if file exists and has correct header
        file_exists = os.path.exists(summary_path)
        if file_exists:
            with open(summary_path, "r", newline="") as existing:
                reader = csv.reader(existing)
                existing_rows = list(reader)
                if existing_rows and existing_rows[0] != header:
                    # Header mismatch - rewrite file with new header
                    rows = existing_rows[1:] if len(existing_rows) > 1 else []
                    with open(summary_path, "w", newline="") as rewritten:
                        writer = csv.writer(rewritten)
                        writer.writerow(header)
                        for row in rows:
                            # Pad or truncate row to match header length
                            padded = row + [""] * max(0, len(header) - len(row))
                            writer.writerow(padded[:len(header)])
        
        # Append new row
        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            algorithm_time = bfs_time_total + dfs_time_total
            writer.writerow([
                graph_folder,
                family,
                graph_name,
                n,
                m,
                f"{total_time:.6f}",
                f"{algorithm_time:.6f}",
                f"{bfs_time_total:.6f}",
                f"{dfs_time_total:.6f}",
                num_iterations,
                num_augmenting_paths,
                max_flow,
                min_cut_value,
                min_cut_str,
            ])
    
    def write_detailed_iterations(self, graph_name: str) -> None:
        """
        Write detailed per-iteration metrics to performance_iterations.csv.
        
        Args:
            graph_name: Name of the graph file
        """
        detailed_path = os.path.join(self.results_dir, "performance_iterations.csv")
        file_exists = os.path.exists(detailed_path)
        
        with open(detailed_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "graph", "iteration", "flow_added", "total_flow", "path"
                ])
            for metric in self.iteration_metrics:
                path_str = " -> ".join(map(str, metric["path"]))
                writer.writerow([
                    graph_name,
                    metric["iteration"],
                    metric["flow_added"],
                    metric["total_flow"],
                    path_str,
                ])
