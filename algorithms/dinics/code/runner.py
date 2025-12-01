"""
Single graph runner for Dinic's algorithm.

This module runs Dinic's algorithm on a single graph, collects metrics,
and saves execution data for later visualization. NO visualization is generated here.
"""
import argparse
import logging
import os
import sys
import time
from typing import List

from dinics import Dinics
from graph_loader import load_graph, validate_graph
from metrics import MetricsCollector
from data_store import save_graph_execution_data


def setup_logger(results_dir: str) -> logging.Logger:
    """Set up logger for the application."""
    logger = logging.getLogger("dinics")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger

    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "run.log")

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def run_single_graph(
    graph_path: str,
    source: int,
    sink: int,
    family: str = "",
    results_dir: str = None,
    logger: logging.Logger = None,
) -> dict:
    """
    Run Dinic's algorithm on a single graph.
    
    This function:
    1. Loads graph from file
    2. Runs algorithm
    3. Collects and saves metrics
    4. Saves execution data for later visualization
    
    NOTE: Visualization is NOT generated here. Use visualizer_script.py separately.
    
    Args:
        graph_path: Path to graph file
        source: Source vertex index
        sink: Sink vertex index
        family: Graph family name (optional)
        results_dir: Directory for results (auto-detected if None)
        logger: Logger instance (optional)
        
    Returns:
        Dictionary with results: max_flow, min_cut_value, total_time, etc.
    """
    # Auto-detect directories if not provided
    if results_dir is None:
        project_root = os.path.dirname(os.path.dirname(__file__))
        results_dir = os.path.join(project_root, "results")
    
    if logger is None:
        logger = setup_logger(results_dir)
    
    # Detect graph folder from path (e.g., graphs, graphs_v, experiment1, ...)
    project_root = os.path.dirname(os.path.dirname(__file__))
    try:
        rel = os.path.relpath(graph_path, project_root)
        first_part = rel.split(os.sep)[0]
        # If graph file is outside project root, fall back to parent folder name
        if first_part.startswith("..") or first_part == "":
            graph_folder = os.path.basename(os.path.dirname(graph_path)) or "graphs"
        else:
            graph_folder = first_part
    except Exception:
        graph_folder = os.path.basename(os.path.dirname(graph_path)) or "graphs"
    
    # Load graph
    logger.info(f"Loading graph from: {graph_path}")
    num_vertices, num_edges, edges = load_graph(graph_path)
    validate_graph(num_vertices, source, sink)
    
    # Initialize algorithm
    dinics = Dinics(num_vertices, edges, source, sink)
    metrics = MetricsCollector(results_dir)
    
    graph_name = os.path.splitext(os.path.basename(graph_path))[0]
    
    # Track path history for visualization data
    path_history: List[tuple] = []
    current_bfs_phase = 0
    path_count = 0
    
    def path_callback(iteration, path_index, path, flow_added, total_flow, is_new_bfs_phase):
        """Callback to track augmenting paths and record metrics."""
        nonlocal current_bfs_phase, path_count
        path_count += 1
        
        if is_new_bfs_phase:
            current_bfs_phase = iteration
        
        path_history.append((iteration, path, flow_added))
        
        # Record iteration metrics
        bfs_time = dinics.bfs_time_total if is_new_bfs_phase and path_count == 1 else 0.0
        dfs_time = 0.0  # Individual DFS time not tracked per path
        metrics.record_iteration(
            iteration=path_index,
            path=path,
            flow_added=flow_added,
            total_flow=total_flow,
            bfs_time=bfs_time,
            dfs_time=dfs_time
        )
    
    # Run algorithm
    # total_time = wall-clock time from start to finish (includes all overhead)
    logger.info(f"Running Dinic's algorithm: source={source}, sink={sink}")
    start_time = time.perf_counter()
    max_flow, path_history_raw = dinics.run(callback=path_callback)
    total_time = time.perf_counter() - start_time
    # Note: total_time is the complete wall-clock runtime.
    # bfs_time_total + dfs_time_total gives the algorithm-only time (may be slightly less due to overhead)
    
    # Get min cut
    min_cut_value, S, T, min_cut_edges = dinics.get_min_cut()
    
    logger.info(f"Maximum flow: {max_flow}")
    logger.info(f"Minimum cut value: {min_cut_value}")
    logger.info(f"Total runtime: {total_time:.6f}s (BFS: {dinics.bfs_time_total:.6f}s, DFS: {dinics.dfs_time_total:.6f}s)")
    
    # Get final state
    final_flow_dist = dinics.get_flow_distribution()
    dinics._bfs_level_graph()  # Get final levels
    
    # Save execution data for later visualization (organized by folder under results/)
    initial_edges = [(u, v, c) for u, v, c in edges]
    json_path = save_graph_execution_data(
        graph_name=graph_name,
        num_vertices=num_vertices,
        source=source,
        sink=sink,
        initial_edges=initial_edges,
        final_flow_dist=final_flow_dist,
        path_history=path_history_raw,
        final_levels=dinics.level,
        results_dir=results_dir,
        graph_folder=graph_folder,
    )
    logger.info(f"Saved execution data: {json_path}")
    # Execution JSON saved under results; no graph_store copy is required.
    # Write metrics
    metrics.write_performance_summary(
        graph_folder=graph_folder,
        graph_name=graph_name,
        family=family,
        n=num_vertices,
        m=num_edges,
        max_flow=max_flow,
        min_cut_value=min_cut_value,
        min_cut_edges=min_cut_edges,
        total_time=total_time,
        bfs_time_total=dinics.bfs_time_total,
        dfs_time_total=dinics.dfs_time_total,
        num_iterations=dinics.bfs_phases,
        num_augmenting_paths=dinics.total_augmenting_paths
    )
    
    metrics.write_detailed_iterations(os.path.basename(graph_path))
    
    # Prepare output summary for global results
    output_path = os.path.join(results_dir, "output.txt")
    algorithm_time = dinics.bfs_time_total + dinics.dfs_time_total

    output_lines = []
    output_lines.append(f"{'='*60}\n")
    output_lines.append(f"Graph: {graph_name}\n")
    output_lines.append(f"Family: {family}\n")
    output_lines.append(f"{'='*60}\n\n")
    output_lines.append(f"Maximum Flow: {max_flow}\n\n")
    output_lines.append("Flow Distribution:\n")
    for u, v, flow, cap in final_flow_dist:
        output_lines.append(f"{u} → {v} : {flow} / {cap}\n")
    output_lines.append(f"\nMinimum Cut:\n")
    output_lines.append(f"S = {sorted(S)}\n")
    output_lines.append(f"T = {sorted(T)}\n")
    output_lines.append("Cut edges:\n")
    for u, v in min_cut_edges:
        cap = next((c for u1, v1, c in edges if u1 == u and v1 == v), 0)
        output_lines.append(f"{u} → {v} (capacity {cap})\n")
    output_lines.append(f"Cut Value = {min_cut_value}\n")
    output_lines.append(f"\nRuntime:\n")
    output_lines.append(f"  Total time: {total_time:.6f}s\n")
    output_lines.append(f"  Algorithm time (BFS+DFS): {algorithm_time:.6f}s\n")
    output_lines.append(f"  BFS time: {dinics.bfs_time_total:.6f}s\n")
    output_lines.append(f"  DFS time: {dinics.dfs_time_total:.6f}s\n")
    output_lines.append(f"\n")

    # Write to global results/output.txt (append)
    try:
        with open(output_path, "a") as f:
            f.writelines(output_lines)
    except Exception:
        logger.warning(f"Could not write to global output.txt for {graph_name}")

    # Per-graph run logs are written to the global results directory and execution JSON; graph_store feature removed.

    return {
        "max_flow": max_flow,
        "min_cut_value": min_cut_value,
        "min_cut_edges": min_cut_edges,
        "S": S,
        "T": T,
        "total_time": total_time,
        "bfs_time_total": dinics.bfs_time_total,
        "dfs_time_total": dinics.dfs_time_total,
        "num_iterations": dinics.bfs_phases,
        "num_augmenting_paths": dinics.total_augmenting_paths
    }


def main() -> None:
    """Main entry point for running a single graph."""
    parser = argparse.ArgumentParser(description="Run Dinic's Algorithm on a graph file.")
    parser.add_argument("--graph", required=True, help="Path to graph file (e.g., graphs/sample1.txt)")
    parser.add_argument("--source", type=int, required=True, help="Source vertex id")
    parser.add_argument("--sink", type=int, required=True, help="Sink vertex id")
    parser.add_argument("--family", type=str, default="", help="Graph family name (optional)")
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")
    logger = setup_logger(results_dir)
    
    graph_path = args.graph
    if not os.path.isabs(graph_path):
        graph_path = os.path.join(project_root, graph_path)

    if not os.path.exists(graph_path):
        logger.error(f"Graph file not found: {graph_path}")
        sys.exit(1)

    result = run_single_graph(
        graph_path=graph_path,
        source=args.source,
        sink=args.sink,
        family=args.family,
        results_dir=results_dir,
        logger=logger
    )

    print(f"Maximum Flow: {result['max_flow']}")
    print(f"Minimum Cut Value: {result['min_cut_value']}")
    print(f"Total Runtime: {result['total_time']:.6f}s")
    print(f"  - BFS time: {result['bfs_time_total']:.6f}s")
    print(f"  - DFS time: {result['dfs_time_total']:.6f}s")
    print(f"\nResults saved to {results_dir}")
    print(f"Run 'python3 code/visualizer_script.py' to generate all visuals")


if __name__ == "__main__":
    main()
