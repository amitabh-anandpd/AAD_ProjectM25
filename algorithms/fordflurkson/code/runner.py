"""
Runner for Ford–Fulkerson that mirrors Dinic's `runner.py` behavior and outputs.
"""
import argparse
import logging
import os
import sys
import time
from typing import List

from ff import FordFulkerson
from graph_loader import load_graph, validate_graph
from metrics import MetricsCollector
from data_store import save_graph_execution_data


def setup_logger(results_dir: str) -> logging.Logger:
    logger = logging.getLogger("fordflurkson")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    os.makedirs(results_dir, exist_ok=True)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    return logger


def run_single_graph(graph_path: str, source: int, sink: int, family: str = "", results_dir: str = None, logger: logging.Logger = None, method: str = "dfs") -> dict:
    if results_dir is None:
        project_root = os.path.dirname(os.path.dirname(__file__))
        results_dir = os.path.join(project_root, "results")

    if logger is None:
        logger = setup_logger(results_dir)

    project_root = os.path.dirname(os.path.dirname(__file__))
    try:
        rel = os.path.relpath(graph_path, project_root)
        first_part = rel.split(os.sep)[0]
        graph_folder = first_part if not first_part.startswith("..") else os.path.basename(os.path.dirname(graph_path))
    except Exception:
        graph_folder = os.path.basename(os.path.dirname(graph_path))

    logger.info(f"Loading graph from: {graph_path}")
    num_vertices, num_edges, edges = load_graph(graph_path)
    validate_graph(num_vertices, source, sink)

    ff = FordFulkerson(num_vertices, edges, source, sink, method=method)
    metrics = MetricsCollector(results_dir)

    graph_name = os.path.splitext(os.path.basename(graph_path))[0]

    # Callback to mirror Dinic's callback signature
    def path_callback(iteration, path_index, path, flow_added, total_flow, is_new_phase):
        metrics.record_iteration(iteration=path_index, path=path, flow_added=flow_added, total_flow=total_flow)

    logger.info(f"Running Ford–Fulkerson (method={method}): source={source}, sink={sink}")
    start_time = time.perf_counter()
    max_flow, path_history = ff.run(callback=path_callback)
    total_time = time.perf_counter() - start_time

    min_cut_value, S, T, min_cut_edges = ff.get_min_cut()

    logger.info(f"Maximum flow: {max_flow}")
    logger.info(f"Minimum cut value: {min_cut_value}")

    final_flow_dist = ff.get_flow_distribution()

    # Save execution data
    initial_edges = [(u, v, c) for u, v, c in edges]
    json_path = save_graph_execution_data(
        graph_name=graph_name,
        num_vertices=num_vertices,
        source=source,
        sink=sink,
        initial_edges=initial_edges,
        final_flow_dist=final_flow_dist,
        path_history=path_history,
        final_levels=[],
        results_dir=results_dir,
        graph_folder=graph_folder,
    )
    logger.info(f"Saved execution data: {json_path}")

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
        bfs_time_total=ff.bfs_time_total,
        dfs_time_total=ff.dfs_time_total,
        num_iterations=0,
        num_augmenting_paths=ff.total_augmenting_paths,
    )

    metrics.write_detailed_iterations(os.path.basename(graph_path))

    return {
        "max_flow": max_flow,
        "min_cut_value": min_cut_value,
        "min_cut_edges": min_cut_edges,
        "S": S,
        "T": T,
        "total_time": total_time,
        "bfs_time_total": ff.bfs_time_total,
        "dfs_time_total": ff.dfs_time_total,
        "num_iterations": 0,
        "num_augmenting_paths": ff.total_augmenting_paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Ford–Fulkerson on a graph file.")
    parser.add_argument("--graph", required=True, help="Path to graph file (e.g., graphs/sample1.txt)")
    parser.add_argument("--source", type=int, required=True, help="Source vertex id")
    parser.add_argument("--sink", type=int, required=True, help="Sink vertex id")
    parser.add_argument("--family", type=str, default="", help="Graph family name (optional)")
    parser.add_argument("--method", type=str, default="dfs", choices=["dfs", "bfs"], help="Augmenting path search method")
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
        logger=logger,
        method=args.method,
    )

    print(f"Maximum Flow: {result['max_flow']}")
    print(f"Minimum Cut Value: {result['min_cut_value']}")
    print(f"Total Runtime: {result['total_time']:.6f}s")
    print(f"  - BFS time: {result['bfs_time_total']:.6f}s")
    print(f"  - DFS time: {result['dfs_time_total']:.6f}s")
    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
