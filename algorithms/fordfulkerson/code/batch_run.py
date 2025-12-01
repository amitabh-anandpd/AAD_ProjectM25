"""
Batch runner for Ford–Fulkerson; mirrors Dinic's `batch_run.py` behavior.
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
    if "layered" in filename_lower:
        return "layered"
    if "crosslinked" in filename_lower:
        return "crosslinked"
    if "dense" in filename_lower:
        return "dense"
    if "sparse" in filename_lower:
        return "sparse"
    if "bidirectional" in filename_lower:
        return "bidirectional"
    return ""


def detect_source_sink_from_graph(graph_path: str) -> tuple:
    with open(graph_path, "r") as f:
        first_line = f.readline().strip().split()
        if len(first_line) >= 1:
            num_vertices = int(first_line[0])
            return 0, num_vertices - 1
    return 0, 0


def process_graph_folder(graph_folder_path: str, graph_folder_name: str, results_dir: str, logger: logging.Logger) -> int:
    graph_files = [f for f in os.listdir(graph_folder_path) if f.endswith('.txt')]
    if not graph_files:
        logger.warning(f"No graph files found in {graph_folder_name}/")
        return 0

    processed = 0
    for filename in sorted(graph_files):
        graph_path = os.path.join(graph_folder_path, filename)
        source, sink = detect_source_sink_from_graph(graph_path)
        family = detect_family_from_filename(filename)
        logger.info(f"Processing: {filename} (family: {family}, source: {source}, sink: {sink})")
        try:
            result = run_single_graph(graph_path=graph_path, source=source, sink=sink, family=family, results_dir=results_dir)
            if result:
                logger.info(f"✓ Completed: max_flow={result['max_flow']}, min_cut={result['min_cut_value']}, runtime={result['total_time']:.6f}s")
                processed += 1
        except Exception as e:
            logger.error(f"✗ Error processing {filename}: {e}")

    return processed


def main():
    parser = argparse.ArgumentParser(description="Run Ford–Fulkerson on graphs in a batch folder")
    parser.add_argument('batch_folder', help='Batch folder to process (required)')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(__file__))
    folder_path = os.path.join(project_root, args.batch_folder)
    if not os.path.isdir(folder_path):
        print(f"Batch folder not found: {args.batch_folder}/")
        sys.exit(1)

    results_dir = os.path.join(project_root, 'results')
    logger = setup_logger(results_dir)
    logger.info(f"Processing batch folder: {args.batch_folder}")

    processed = process_graph_folder(folder_path, args.batch_folder, results_dir, logger)
    logger.info(f"Total graphs processed: {processed}")


if __name__ == '__main__':
    main()
