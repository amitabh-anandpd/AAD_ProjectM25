"""
Standalone visualizer script for generating graphs on demand.

Usage:
    python3 code/visualizer_script.py [batch_folder]
    
    If batch_folder is specified, generates visuals only for graphs from that batch folder.
    Valid batch folders: graphs, graphs_v, graphs_e
"""
import os
import sys
import argparse

from visualizer import GraphVisualizer
from data_store import find_all_execution_data, load_graph_execution_data
from dinics import Dinics


def generate_visuals_from_data(data: dict, visuals_dir: str, graph_name: str, graph_folder: str):
    """
    Generate all visualizations from stored execution data.
    
    Args:
        data: Dictionary containing execution data
        visuals_dir: Directory to save visuals
        graph_name: Name of the graph
        graph_folder: Name of the graph folder (graphs, graphs2, etc.)
    """
    num_vertices = data["num_vertices"]
    source = data["source"]
    sink = data["sink"]
    initial_edges = data["initial_edges"]
    final_flow_dist = data["final_flow_dist"]
    path_history = data["path_history"]
    final_levels = data["final_levels"]
    
    visualizer = GraphVisualizer(num_vertices, source, sink)
    
    # Organize visuals by folder: visuals/graphs/, visuals/graphs2/, etc.
    folder_visuals_dir = os.path.join(visuals_dir, graph_folder)
    graph_visuals_dir = os.path.join(folder_visuals_dir, graph_name)
    os.makedirs(graph_visuals_dir, exist_ok=True)
    
    # 1. Generate initial graph
    initial_flow_edges = [(u, v, 0, c) for u, v, c in initial_edges]
    initial_levels = visualizer.compute_bfs_levels(initial_flow_edges, source)
    visualizer.visualize_graph(
        edges=initial_flow_edges,
        output_path=os.path.join(graph_visuals_dir, "initial_graph.png"),
        title="Initial Flow Network",
        levels=initial_levels
    )
    
    # 2. Generate iteration visualizations
    for iter_idx, (iteration_num, path, flow_added) in enumerate(path_history, 1):
        iter_dir = os.path.join(graph_visuals_dir, f"iteration_{iter_idx}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # Reconstruct state up to this point
        temp_dinics = Dinics(num_vertices, initial_edges, source, sink)
        path_count = 0
        
        while path_count < iter_idx:
            if not temp_dinics._bfs_level_graph():
                break
            temp_dinics.it_ptr = [0] * num_vertices
            while path_count < iter_idx:
                path_flow, path_nodes = temp_dinics._dfs_find_path(source, float("inf"), [source])
                if path_flow <= 0:
                    break
                temp_dinics._apply_path(path_nodes, path_flow)
                path_count += 1
                if path_count >= iter_idx:
                    break
        
        # Get state before this path
        if iter_idx > 1:
            temp_dinics_before = Dinics(num_vertices, initial_edges, source, sink)
            path_count = 0
            while path_count < iter_idx - 1:
                if not temp_dinics_before._bfs_level_graph():
                    break
                temp_dinics_before.it_ptr = [0] * num_vertices
                while path_count < iter_idx - 1:
                    path_flow, path_nodes = temp_dinics_before._dfs_find_path(source, float("inf"), [source])
                    if path_flow <= 0:
                        break
                    temp_dinics_before._apply_path(path_nodes, path_flow)
                    path_count += 1
                    if path_count >= iter_idx - 1:
                        break
            
            flow_dist_before = temp_dinics_before.get_flow_distribution()
            residual_edges_before = temp_dinics_before.get_residual_edges()
            temp_dinics_before._bfs_level_graph()
            levels_before = temp_dinics_before.level
        else:
            flow_dist_before = initial_flow_edges
            residual_edges_before = [(u, v, c, True) for u, v, c in initial_edges]
            temp_dinics_temp = Dinics(num_vertices, initial_edges, source, sink)
            temp_dinics_temp._bfs_level_graph()
            levels_before = temp_dinics_temp.level
        
        # Get state after this path
        flow_dist = temp_dinics.get_flow_distribution()
        residual_edges = temp_dinics.get_residual_edges()
        temp_dinics._bfs_level_graph()
        levels_after = temp_dinics.level
        
        # Generate initial residual
        visualizer.visualize_graph(
            edges=flow_dist_before,
            output_path=os.path.join(iter_dir, "initial_residual.png"),
            title=f"Iteration {iter_idx}: Residual Network (Before)",
            levels=levels_before,
            residual_edges=residual_edges_before,
            mode="residual"
        )
        
        # Generate augmenting path
        visualizer.visualize_graph(
            edges=flow_dist_before,
            output_path=os.path.join(iter_dir, "selected_augmented_path.png"),
            title=f"Iteration {iter_idx}: Augmenting Path",
            levels=levels_before,
            highlight_path=path
        )
        
        # Generate final residual
        visualizer.visualize_graph(
            edges=flow_dist,
            output_path=os.path.join(iter_dir, "final_residual.png"),
            title=f"Iteration {iter_idx}: Residual Network (After)",
            levels=levels_after,
            residual_edges=residual_edges,
            mode="residual"
        )
    
    # 3. Generate final flow graph (FIXED: recompute levels from final flow graph)
    final_levels_computed = visualizer.compute_bfs_levels(final_flow_dist, source)
    visualizer.visualize_graph(
        edges=final_flow_dist,
        output_path=os.path.join(graph_visuals_dir, "final_flow_graph.png"),
        title="Final Max Flow Network",
        levels=final_levels_computed,
        mode="full"
    )
    
    # Histograms removed - not needed


def main():
    """
    Main entry point: Generate visualizations for execution data files.
    
    Optionally filters by batch folder if specified.
    """
    parser = argparse.ArgumentParser(
        description="Generate visualizations from execution data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 code/visualizer_script.py graphs      # Visualize only graphs/ batch
  python3 code/visualizer_script.py graphs_v    # Visualize only graphs_v/ batch
  python3 code/visualizer_script.py graphs_e    # Visualize only graphs_e/ batch
  python3 code/visualizer_script.py             # Visualize all batches
        """
    )
    parser.add_argument(
        "batch_folder",
        nargs="?",
        choices=["graphs", "graphs_v", "graphs_e"],
        help="Batch folder to visualize (graphs, graphs_v, or graphs_e). If not specified, visualizes all."
    )
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")
    visuals_dir = os.path.join(project_root, "visuals")
    
    # Find all execution data files
    execution_files = find_all_execution_data(results_dir)
    
    # Filter by batch folder if specified
    if args.batch_folder:
        execution_files = [(gf, jp) for gf, jp in execution_files if gf == args.batch_folder]
    
    if not execution_files:
        if args.batch_folder:
            print(f"No execution data files found for batch folder: {args.batch_folder}")
            print("Please run the algorithm first using:")
            print(f"  python3 code/batch_run.py {args.batch_folder}")
        else:
            print("No execution data files found.")
            print("Please run the algorithm first using:")
            print("  python3 code/batch_run.py")
            print("  or")
            print("  python3 code/runner.py --graph <graph_file> --source <s> --sink <t>")
        sys.exit(1)
    
    batch_label = f" ({args.batch_folder})" if args.batch_folder else ""
    print(f"Found {len(execution_files)} execution data file(s){batch_label}")
    print("Generating visualizations...\n")
    
    # Process each execution data file
    for graph_folder, json_path in execution_files:
        print(f"Processing: {graph_folder}/{os.path.basename(json_path)}")
        
        try:
            data = load_graph_execution_data(json_path)
            graph_name = data["graph_name"]
            graph_folder_name = data.get("graph_folder", graph_folder)
            
            generate_visuals_from_data(data, visuals_dir, graph_name, graph_folder_name)
            print(f"  ✓ Generated visuals for {graph_name}\n")
            
        except Exception as e:
            print(f"  ✗ Error processing {json_path}: {e}\n")
    
    print(f"✓ All visualizations generated in {visuals_dir}")
    if args.batch_folder:
        print(f"Organized by folder: visuals/{args.batch_folder}/")
    else:
        print(f"Organized by folder: visuals/graphs/, visuals/graphs_v/, visuals/graphs_e/, etc.")


if __name__ == "__main__":
    main()
