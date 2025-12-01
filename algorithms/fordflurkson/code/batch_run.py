"""
Batch runner for Ford–Fulkerson experiments (adapted from Dinic's batch_run).

Usage:
  python3 code/batch_run.py <batch_folder>

This will process all `.txt` graphs in the specified folder and run the
Ford–Fulkerson runner using the appropriate traversal method.
"""
import os
import sys
import argparse
import logging

from runner import run_single_graph, setup_logger


def detect_family_from_filename(filename: str) -> str:
    filename_lower = filename.lower()
    if filename_lower.startswith("sample"):
        return "sample"
    elif "layered" in filename_lower:
        return "layered"
    elif "crosslinked" in filename_lower:
        return "crosslinked"
    elif "dense" in filename_lower:
        return "dense"
    elif "sparse" in filename_lower:
        return "sparse"
    elif "bidirectional" in filename_lower:
        return "bidirectional"
    elif filename_lower.startswith("type_v"):
        return "type_v"
    elif filename_lower.startswith("type_e"):
        return "type_e"
    return ""


def detect_source_sink_from_graph(graph_path: str) -> tuple:
    with open(graph_path, "r") as f:
        first_line = f.readline().strip().split()
        if len(first_line) >= 1:
            num_vertices = int(first_line[0])
            return 0, num_vertices - 1
    return 0, 0


def process_graph_folder(graph_folder_path: str, graph_folder_name: str, results_dir: str, logger: logging.Logger) -> int:
    sample_configs = {
        "sample1.txt": (0, 5),
        "sample2.txt": (0, 7),
        "sample3.txt": (0, 9),
    }

    family_configs = {
        "layered_1.txt": (0, 5),
        "layered_2.txt": (0, 7),
        "layered_3.txt": (0, 9),
        "crosslinked_1.txt": (0, 6),
        "crosslinked_2.txt": (0, 7),
        "crosslinked_3.txt": (0, 9),
        "dense_1.txt": (0, 5),
        "dense_2.txt": (0, 7),
        "dense_3.txt": (0, 9),
        "sparse_1.txt": (0, 5),
        "sparse_2.txt": (0, 7),
        "sparse_3.txt": (0, 9),
        "bidirectional_1.txt": (0, 5),
        "bidirectional_2.txt": (0, 7),
        "bidirectional_3.txt": (0, 9),
    }

    all_configs = {**sample_configs, **family_configs}
    graph_files = [f for f in os.listdir(graph_folder_path) if f.endswith(".txt")]

    if not graph_files:
        logger.warning(f"No graph files found in {graph_folder_name}/")
        return 0

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing folder: {graph_folder_name}/ ({len(graph_files)} graph(s))")
    logger.info(f"{'='*60}")

    processed = 0
    for filename in sorted(graph_files):
        graph_path = os.path.join(graph_folder_path, filename)
        if filename in all_configs:
            source, sink = all_configs[filename]
        else:
            source, sink = detect_source_sink_from_graph(graph_path)
            logger.warning(f"Using default source={source}, sink={sink} for {filename}")

        family = detect_family_from_filename(filename)
        logger.info(f"\nProcessing: {filename} (family: {family}, source: {source}, sink: {sink})")

        # For every input graph we run both traversal methods so that results
        # are produced separately for 'classical' (DFS) and 'edmondson' (BFS).
        methods = [("dfs", "classical"), ("bfs", "edmondson")]
        for method_key, method_label in methods:
            try:
                result = run_single_graph(
                    graph_path=graph_path,
                    source=source,
                    sink=sink,
                    method=method_key,
                    family=family,
                    results_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", method_label),
                    logger=logger
                )

                if result:
                    logger.info(f"✓ [{method_label}] Completed: max_flow={result['max_flow']}, min_cut={result['min_cut_value']}, runtime={result['total_time']:.6f}s")
                    # Only count once per graph (regardless of method)
                else:
                    logger.warning(f"✗ [{method_label}] Failed to process {filename}")
            except Exception as e:
                logger.error(f"✗ [{method_label}] Error processing {filename}: {e}", exc_info=True)
        processed += 1

    return processed


def main():
    parser = argparse.ArgumentParser(description="Run Ford–Fulkerson on graphs in a batch folder")
    parser.add_argument("batch_folder", help="Batch folder to process (required)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")

    folder_path = os.path.join(project_root, args.batch_folder)
    if not os.path.isdir(folder_path):
        print(f"Batch folder not found: {args.batch_folder}/")
        sys.exit(1)

    logger = setup_logger(results_dir)
    batch_folders = [(args.batch_folder, folder_path)]

    logger.info("=" * 60)
    logger.info("Starting batch experiments")
    logger.info(f"Processing batch folder: {batch_folders[0][0]}")
    logger.info("=" * 60)

    total_processed = 0
    for folder_name, folder_path in batch_folders:
        processed = process_graph_folder(folder_path, folder_name, results_dir, logger)
        total_processed += processed

    logger.info("\n" + "=" * 60)
    logger.info("Batch experiments complete!")
    logger.info("=" * 60)
    logger.info(f"Total graphs processed: {total_processed}")
    logger.info(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
