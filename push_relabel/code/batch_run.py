"""
Batch runner for multiple graph experiments.
"""
import os
import logging
from runner import setup_logger
from push_relabel import PushRelabel


def run_graph(
    graph_path: str, source: int, sink: int, 
    family: str = "", seed: int = 0, trial: int = 0, 
    logger: logging.Logger = None
):
    """Run Push-Relabel algorithm on a single graph."""
    if not os.path.exists(graph_path):
        if logger:
            logger.error(f"Graph file not found: {graph_path}")
        return None
    
    pr = PushRelabel(graph_path=graph_path, source=source, sink=sink, logger=logger)
    max_flow, runtime = pr.run()
    
    graph_name = os.path.basename(graph_path)
    pr.write_outputs(max_flow=max_flow, runtime=runtime)
    pr.append_metrics(
        graph_name=graph_name, max_flow=max_flow, runtime=runtime,
        family=family, seed=seed, trial=trial
    )
    
    return max_flow, runtime


def main():
    """Run experiments on all sample and family graphs."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")
    graphs_dir = os.path.join(project_root, "graphs")
    logger = setup_logger(results_dir)
    
    # Sample graphs configuration
    samples = [
        ("sample1.txt", 0, 5, "sample"),
        ("sample2.txt", 0, 7, "sample"),
        ("sample3.txt", 0, 9, "sample"),
    ]
    
    # Family graphs configuration
    families = [
        ("family_A_layered.txt", 0, 5, "layered"),
        ("family_B_crosslinked.txt", 0, 5, "crosslinked"),
        ("family_C_dense.txt", 0, 5, "dense"),
        ("family_D_sparse.txt", 0, 5, "sparse"),
        ("family_E_bidirectional.txt", 0, 5, "bidirectional"),
    ]
    
    logger.info("=" * 60)
    logger.info("Starting Push-Relabel batch experiments")
    logger.info("=" * 60)
    
    # Run sample graphs
    logger.info("\n--- Running Sample Graphs ---")
    for filename, source, sink, family in samples:
        graph_path = os.path.join(graphs_dir, filename)
        logger.info(f"\nProcessing: {filename} (source={source}, sink={sink})")
        try:
            result = run_graph(graph_path, source, sink, family=family, logger=logger)
            if result:
                max_flow, runtime = result
                logger.info(f"  ✓ Completed: max_flow={max_flow}, runtime={runtime:.6f}s")
            else:
                logger.warning(f"  ✗ Failed to process {filename}")
        except Exception as e:
            logger.error(f"  ✗ Error processing {filename}: {e}")
    
    # Run family graphs
    logger.info("\n--- Running Family Graphs ---")
    for filename, source, sink, family in families:
        graph_path = os.path.join(graphs_dir, filename)
        logger.info(f"\nProcessing: {filename} (family={family})")
        try:
            result = run_graph(graph_path, source, sink, family=family, logger=logger)
            if result:
                max_flow, runtime = result
                logger.info(f"  ✓ Completed: max_flow={max_flow}, runtime={runtime:.6f}s")
            else:
                logger.warning(f"  ✗ Failed to process {filename}")
        except Exception as e:
            logger.error(f"  ✗ Error processing {filename}: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Batch experiments complete!")
    logger.info("=" * 60)
    logger.info(f"\nResults saved to: {results_dir}")
    logger.info(f"Run 'python code/analyze.py' to generate analysis plots")


if __name__ == "__main__":
    main()
