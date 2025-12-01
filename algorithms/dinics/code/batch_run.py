"""
Batch runner for multiple graph experiments.

This module runs Dinic's algorithm on graphs in a specified batch folder:
graphs/, graphs_v/, or graphs_e/.

Usage:
    python3 code/batch_run.py [batch_folder]
    
    If batch_folder is not specified, processes all batch folders.
    Valid batch folders: graphs, graphs_v, graphs_e
"""
import os
import sys
import argparse
import logging

from runner import run_single_graph, setup_logger


def detect_family_from_filename(filename: str) -> str:
    """
    Auto-detect graph family from filename.
    
    Args:
        filename: Graph filename
        
    Returns:
        Detected family name, or empty string if unknown
    """
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
    """
    Detect source and sink from graph structure.
    
    Args:
        graph_path: Path to graph file
        
    Returns:
        Tuple of (source, sink) vertex indices
    """
    with open(graph_path, "r") as f:
        first_line = f.readline().strip().split()
        if len(first_line) >= 1:
            num_vertices = int(first_line[0])
            return 0, num_vertices - 1
    
    return 0, 0


def find_batch_folders(project_root: str) -> list:
    """
    Find the main batch graph folders: graphs/, graphs_v/, graphs_e/.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        List of (folder_name, folder_path) tuples for batch folders that exist
    """
    batch_folders = []
    
    # Main batch folders
    batch_folder_names = ["graphs", "graphs_v", "graphs_e"]
    
    for folder_name in batch_folder_names:
        folder_path = os.path.join(project_root, folder_name)
        if os.path.isdir(folder_path):
            batch_folders.append((folder_name, folder_path))
    
    return batch_folders


def process_graph_folder(graph_folder_path: str, graph_folder_name: str, 
                        results_dir: str, logger: logging.Logger) -> int:
    """
    Process all graphs in a single folder.
    
    Args:
        graph_folder_path: Path to the graph folder
        graph_folder_name: Name of the folder (graphs, graphs2, etc.)
        results_dir: Directory for results
        logger: Logger instance
        
    Returns:
        Number of graphs processed successfully
    """
    # Sample graphs configuration: (filename, source, sink)
    sample_configs = {
        "sample1.txt": (0, 5),
        "sample2.txt": (0, 7),
        "sample3.txt": (0, 9),
    }
    
    # Family graphs configuration: (filename, source, sink)
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
    
    # Get all .txt files in this folder
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
        
        # Get source and sink
        if filename in all_configs:
            source, sink = all_configs[filename]
        else:
            source, sink = detect_source_sink_from_graph(graph_path)
            logger.warning(f"Using default source={source}, sink={sink} for {filename}")
        
        # Detect family
        family = detect_family_from_filename(filename)
        
        logger.info(f"\nProcessing: {filename} (family: {family}, source: {source}, sink: {sink})")
        
        try:
            result = run_single_graph(
                graph_path=graph_path,
                source=source,
                sink=sink,
                family=family,
                results_dir=results_dir,
                logger=logger
            )
            
            if result:
                logger.info(f"✓ Completed: max_flow={result['max_flow']}, "
                          f"min_cut={result['min_cut_value']}, "
                          f"runtime={result['total_time']:.6f}s")
                processed += 1
            else:
                logger.warning(f"✗ Failed to process {filename}")
                
        except Exception as e:
            logger.error(f"✗ Error processing {filename}: {e}", exc_info=True)
    
    return processed


def main():
    """
    Run experiments on graphs in a specified batch folder or all batch folders.
    
    This function:
    1. Accepts optional batch folder argument (graphs, graphs_v, graphs_e)
    2. Processes the specified folder (or all if not specified)
    3. Saves execution data organized by folder
    
    Note: Cleanup is done manually by the user (rm -rf results/ visuals/ plots/)
    """
    parser = argparse.ArgumentParser(
        description="Run Dinic's algorithm on graphs in a batch folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 code/batch_run.py graphs      # Process only graphs/ folder
  python3 code/batch_run.py graphs_v    # Process only graphs_v/ folder
  python3 code/batch_run.py graphs_e    # Process only graphs_e/ folder
  python3 code/batch_run.py             # Process all batch folders
        """
    )
    parser.add_argument(
        "batch_folder",
        help="Batch folder to process (e.g., graphs, graphs_v, graphs_e, or any custom folder). This argument is required."
    )
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")
    
    # Batch folder is required now — operate only on a single explicit batch
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
    
    # Process each batch folder
    for folder_name, folder_path in batch_folders:
        processed = process_graph_folder(
            folder_path, folder_name, results_dir, logger
        )
        total_processed += processed
    
    logger.info("\n" + "=" * 60)
    logger.info("Batch experiments complete!")
    logger.info("=" * 60)
    logger.info(f"Total graphs processed: {total_processed}")
    logger.info(f"\nResults saved to: {results_dir}")
    logger.info(f"Execution data organized by folder: results/graphs/, results/graphs_v/, results/graphs_e/")
    logger.info(f"\nTo generate visualizations for a specific batch, run:")
    logger.info(f"  python3 code/visualizer_script.py <batch_folder>")
    logger.info(f"\nTo generate analysis table for a specific batch, run:")
    logger.info(f"  python3 code/analyze.py <batch_folder>")
    logger.info(f"\nTo generate scatter plots for a specific batch, run:")
    logger.info(f"  python3 code/plot.py <batch_folder>")


if __name__ == "__main__":
    main()
