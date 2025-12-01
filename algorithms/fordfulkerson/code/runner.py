"""
Batch/single-run runner for Ford–Fulkerson implementations.

Supports two traversal modes:
 - "dfs" (classical Ford–Fulkerson)
 - "bfs" (Edmonds–Karp)

This mirrors the features of `algorithms/dinics/code/runner.py`:
 - loads a single graph file
 - runs the algorithm with chosen traversal method
 - collects metrics and saves execution JSONs via `data_store.save_graph_execution_data`
 - writes performance summaries into `results/<method>/`

Usage:
  python3 code/runner.py --graph <path> --source 0 --sink N-1 --method bfs

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
    method: str = "dfs",
    family: str = "",
    results_dir: str = None,
    logger: logging.Logger = None,
):
    if results_dir is None:
        project_root = os.path.dirname(os.path.dirname(__file__))
        results_dir = os.path.join(project_root, "results", method)

    if logger is None:
        logger = setup_logger(results_dir)

    # Determine graph folder
    project_root = os.path.dirname(os.path.dirname(__file__))
    try:
        rel = os.path.relpath(graph_path, project_root)
        first_part = rel.split(os.sep)[0]
        if first_part.startswith("..") or first_part == "":
            graph_folder = os.path.basename(os.path.dirname(graph_path)) or "graphs"
        else:
            graph_folder = first_part
    except Exception:
        graph_folder = os.path.basename(os.path.dirname(graph_path)) or "graphs"

    logger.info(f"Loading graph: {graph_path}")
    num_vertices, num_edges_declared, edges = load_graph(graph_path)
    validate_graph(num_vertices, source, sink)

    # Initialize algorithm
    ff = FordFulkerson(num_vertices, edges, source, sink, method=method)
    metrics = MetricsCollector(results_dir)

    graph_name = os.path.splitext(os.path.basename(graph_path))[0]

    # Path history will be produced by ff.run
    def callback(iteration, path_index, path, flow_added, total_flow, is_new_bfs_phase):
        # For Ford–Fulkerson we simply record iteration metrics compatible with Dinic's runner
        metrics.record_iteration(
            iteration=path_index,
            path=path,
            flow_added=flow_added,
            total_flow=total_flow,
            bfs_time=ff.bfs_time_total,
            dfs_time=ff.dfs_time_total,
        )

    logger.info(f"Running Ford–Fulkerson ({method}) on graph: source={source}, sink={sink}")
    start_time = time.perf_counter()
    max_flow, path_history = ff.run(callback=callback)
    total_time = time.perf_counter() - start_time

    min_cut_value, S, T, min_cut_edges = ff.get_min_cut()

    logger.info(f"Max flow: {max_flow}  min_cut: {min_cut_value}  time: {total_time:.6f}s")

    final_flow_dist = ff.get_flow_distribution()


    # Save execution JSON under results/<method>/<graph_folder>/
    results_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    method_results_dir = os.path.join(results_root, method)
    json_path = save_graph_execution_data(
        graph_name=graph_name,
        num_vertices=num_vertices,
        source=source,
        sink=sink,
        initial_edges=edges,
        final_flow_dist=final_flow_dist,
        path_history=path_history,
        final_levels=[],
        results_dir=method_results_dir,
        graph_folder=graph_folder,
    )
    logger.info(f"Saved execution data: {json_path}")

    # Also mirror results into user-friendly partitions for visualization/analysis:
    # map method -> partition name: dfs -> classical, bfs -> edmondson
    partition_name = "classical" if method == "dfs" else "edmondson"
    partition_results_dir = os.path.join(results_root, partition_name)

    # Save a second copy of the execution data under the partition folder
    json_path_partition = save_graph_execution_data(
        graph_name=graph_name,
        num_vertices=num_vertices,
        source=source,
        sink=sink,
        initial_edges=edges,
        final_flow_dist=final_flow_dist,
        path_history=path_history,
        final_levels=[],
        results_dir=partition_results_dir,
        graph_folder=graph_folder,
    )
    logger.info(f"Also saved partition execution data: {json_path_partition}")

    # For compatibility with Dinic tooling, also save execution JSON under
    # results/<graph_folder>/ so visualizer and analysis scripts that expect
    # results organized by batch folder can find execution JSONs.
    # Save under the top-level results root and let save_graph_execution_data
    # organize into results/<graph_folder>/ to match Dinic layout.
    json_path_batch = save_graph_execution_data(
        graph_name=graph_name,
        num_vertices=num_vertices,
        source=source,
        sink=sink,
        initial_edges=edges,
        final_flow_dist=final_flow_dist,
        path_history=path_history,
        final_levels=[],
        results_dir=results_root,
        graph_folder=graph_folder,
    )
    logger.info(f"Also saved batch execution data: {json_path_batch}")

    # Convert cut edge tuples from (u,v,c) -> (u,v) for CSV field
    min_cut_edges_uv = [(u, v) for u, v, _ in min_cut_edges]

    # Write performance summary to both method-specific and partition summaries
    metrics.write_performance_summary(
        graph_name=graph_name,
        graph_folder=graph_folder,
        family=family,
        n=num_vertices,
        m=num_edges_declared,
        max_flow=max_flow,
        min_cut_value=min_cut_value,
        min_cut_edges=min_cut_edges_uv,
        total_time=total_time,
        bfs_time_total=ff.bfs_time_total,
        dfs_time_total=ff.dfs_time_total,
        num_iterations=0,
        num_augmenting_paths=ff.total_augmenting_paths,
    )

    # Also append to partition summary CSV (create a separate MetricsCollector)
    partition_metrics = MetricsCollector(partition_results_dir)
    partition_metrics.iteration_metrics = metrics.iteration_metrics
    partition_metrics.write_performance_summary(
        graph_name=graph_name,
        graph_folder=graph_folder,
        family=family,
        n=num_vertices,
        m=num_edges_declared,
        max_flow=max_flow,
        min_cut_value=min_cut_value,
        min_cut_edges=min_cut_edges_uv,
        total_time=total_time,
        bfs_time_total=ff.bfs_time_total,
        dfs_time_total=ff.dfs_time_total,
        num_iterations=0,
        num_augmenting_paths=ff.total_augmenting_paths,
    )

    metrics.write_detailed_iterations(graph_name)
    partition_metrics.write_detailed_iterations(graph_name)

    return {
        "max_flow": max_flow,
        "min_cut_value": min_cut_value,
        "total_time": total_time,
        "bfs_time_total": ff.bfs_time_total,
        "dfs_time_total": ff.dfs_time_total,
        "num_augmenting_paths": ff.total_augmenting_paths,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Ford–Fulkerson on a graph file.")
    parser.add_argument("--graph", required=True, help="Path to graph file")
    parser.add_argument("--source", type=int, required=True, help="Source vertex id")
    parser.add_argument("--sink", type=int, required=True, help="Sink vertex id")
    parser.add_argument("--method", choices=["dfs", "bfs"], default="dfs",
                        help="Traversal mode: dfs=classical Ford–Fulkerson, bfs=Edmonds–Karp")
    parser.add_argument("--family", type=str, default="", help="Graph family name (optional)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(__file__))
    graph_path = args.graph
    # Accept either an absolute path, a path relative to current working dir,
    # or a path relative to the project root (algorithms/...). Try all.
    candidates = []
    if os.path.isabs(graph_path):
        candidates.append(graph_path)
    candidates.append(os.path.join(os.getcwd(), graph_path))
    candidates.append(os.path.join(project_root, graph_path))

    resolved = None
    for p in candidates:
        if os.path.exists(p):
            resolved = p
            break

    if resolved is None:
        print(f"Graph not found (tried {candidates}): {graph_path}")
        sys.exit(1)

    graph_path = resolved

    results_dir = os.path.join(project_root, "results", args.method)
    logger = setup_logger(results_dir)

    res = run_single_graph(
        graph_path=graph_path,
        source=args.source,
        sink=args.sink,
        method=args.method,
        family=args.family,
        results_dir=results_dir,
        logger=logger,
    )

    print(f"Max flow: {res['max_flow']}")
    print(f"Min cut value: {res['min_cut_value']}")
    print(f"Total time: {res['total_time']:.6f}s")


if __name__ == "__main__":
    main()
