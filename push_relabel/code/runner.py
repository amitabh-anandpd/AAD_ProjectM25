"""
CLI runner for Push-Relabel algorithm on single graphs.
"""
import argparse
import logging
import os
import sys

from push_relabel import PushRelabel


def setup_logger(results_dir: str) -> logging.Logger:
    """Setup logger for console and file output."""
    logger = logging.getLogger("push_relabel")
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


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(description="Run Push-Relabel Algorithm on a graph file.")
    parser.add_argument("--graph", required=True, help="Path to graph file")
    parser.add_argument("--source", type=int, required=True, help="Source vertex id")
    parser.add_argument("--sink", type=int, required=True, help="Sink vertex id")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")
    logger = setup_logger(results_dir)

    graph_path = args.graph
    if not os.path.isabs(graph_path):
        graph_path = os.path.join(project_root, graph_path)

    if not os.path.exists(graph_path):
        logger.error("Graph file not found: %s", graph_path)
        sys.exit(1)

    pr = PushRelabel(graph_path=graph_path, source=args.source, sink=args.sink, logger=logger)
    max_flow, runtime = pr.run()

    # Write outputs and metrics
    graph_name = os.path.basename(graph_path)
    pr.write_outputs(max_flow=max_flow, runtime=runtime)
    pr.append_metrics(graph_name=graph_name, max_flow=max_flow, runtime=runtime)

    print(f"\n{'='*60}")
    print(f"Maximum Flow: {max_flow}")
    print(f"Runtime: {runtime:.6f}s")
    print(f"Push Operations: {pr.num_pushes}")
    print(f"Relabel Operations: {pr.num_relabels}")
    print(f"{'='*60}")
    print(f"\nResults written to: {os.path.join(results_dir, 'output.txt')}")
    print(f"Visualizations: {pr.graph_visuals_dir}")


if __name__ == "__main__":
    main()
