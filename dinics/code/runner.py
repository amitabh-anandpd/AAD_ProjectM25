import argparse
import logging
import os
import sys

from dinics import Dinics


def setup_logger(results_dir: str) -> logging.Logger:
    logger = logging.getLogger("dinics")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if run multiple times in same interpreter
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
    parser = argparse.ArgumentParser(description="Run Dinic's Algorithm on a graph file.")
    parser.add_argument("--graph", required=True, help="Path to graph file (e.g., graphs/sample1.txt)")
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

    dinic = Dinics(graph_path=graph_path, source=args.source, sink=args.sink, logger=logger)
    max_flow, runtime = dinic.run()

    # Output and metrics
    graph_name = os.path.basename(graph_path)
    dinic.write_outputs(max_flow=max_flow, runtime=runtime)
    dinic.append_metrics(graph_name=graph_name, max_flow=max_flow, runtime=runtime)
    dinic.write_iteration_metrics(graph_name=graph_name)

    print(f"Maximum Flow: {max_flow}")
    print(f"Runtime: {runtime:.4f}s")
    print(f"Wrote results to {os.path.join(results_dir, 'output.txt')}")
    print(f"Iteration visuals: {os.path.join(project_root, 'visuals')}")
    print(f"Detailed metrics appended to {os.path.join(results_dir, 'performance_iterations.csv')}")


if __name__ == "__main__":
    main()


