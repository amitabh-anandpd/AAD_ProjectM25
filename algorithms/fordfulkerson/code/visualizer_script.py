"""
Visualizer script for Ford–Fulkerson that generates visuals from saved execution JSONs.

This version reconstructs per-iteration flow state by applying the stored
`path_history` from the execution data to the `initial_edges` (no need to
recreate the algorithm internals).

Usage:
  python3 code/visualizer_script.py [batch_folder]

If `batch_folder` specified, will only process execution JSONs for that folder.
"""
import os
import sys
import argparse
from visualizer import GraphVisualizer
from data_store import find_all_execution_data, load_graph_execution_data


def apply_path_to_flow(initial_flow: dict, path: list, flow: int) -> None:
    """
    Apply an augmenting `path` with `flow` to the flow dictionary in-place.

    initial_flow: mapping (u,v) -> (flow, capacity)
    path: list of node ids (e.g., [0,3,5])
    """
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if (u, v) in initial_flow:
            # forward edge exists
            f, c = initial_flow[(u, v)]
            initial_flow[(u, v)] = (f + flow, c)
        elif (v, u) in initial_flow:
            # push on reverse of an original edge
            f, c = initial_flow[(v, u)]
            initial_flow[(v, u)] = (f - flow, c)
        else:
            # Edge may be a residual-only edge; add it as forward with capacity inf
            initial_flow[(u, v)] = (flow, 0)


def flow_dict_to_edge_list(flow_dict: dict, n: int) -> list:
    """Convert mapping (u,v)->(flow,cap) to list of (u,v,flow,cap) for visualizer."""
    edges = []
    for (u, v), (f, c) in flow_dict.items():
        edges.append((u, v, int(f), int(c)))
    # ensure all nodes exist even if isolated
    return edges


def generate_visuals_from_data(data: dict, visuals_dir: str, graph_name: str, graph_folder: str):
    num_vertices = data["num_vertices"]
    source = data["source"]
    sink = data["sink"]
    initial_edges = data["initial_edges"]  # list of [u,v,c]
    path_history = data.get("path_history", [])
    final_flow_dist = data.get("final_flow_dist", [])

    visualizer = GraphVisualizer(num_vertices, source, sink)

    folder_visuals_dir = os.path.join(visuals_dir, graph_folder)
    graph_visuals_dir = os.path.join(folder_visuals_dir, graph_name)
    os.makedirs(graph_visuals_dir, exist_ok=True)

    # Build initial flow dict (u,v) -> (flow, capacity)
    flow_map = {}
    for u, v, c in initial_edges:
        flow_map[(u, v)] = (0, c)

    # Save initial graph image
    initial_flow_edges = flow_dict_to_edge_list(flow_map, num_vertices)
    levels_init = visualizer.compute_bfs_levels(initial_flow_edges, source)
    visualizer.visualize_graph(edges=initial_flow_edges, output_path=os.path.join(graph_visuals_dir, "initial_graph.png"), title="Initial Flow Network", levels=levels_init)

    # Iterate and create visuals per augmenting path
    for idx, (iteration_num, path, flow_added) in enumerate(path_history, 1):
        iter_dir = os.path.join(graph_visuals_dir, f"iteration_{idx}")
        os.makedirs(iter_dir, exist_ok=True)

        # State before applying this path: recreate from scratch applying previous paths
        before_flow = {}
        for u, v, c in initial_edges:
            before_flow[(u, v)] = (0, c)

        for j in range(idx - 1):
            _, pth, fadded = path_history[j]
            apply_path_to_flow(before_flow, pth, fadded)

        # State after applying this path
        after_flow = {k: v for k, v in before_flow.items()}
        apply_path_to_flow(after_flow, path, flow_added)

        flow_dist_before = flow_dict_to_edge_list(before_flow, num_vertices)
        flow_dist_after = flow_dict_to_edge_list(after_flow, num_vertices)

        levels_before = visualizer.compute_bfs_levels(flow_dist_before, source)
        levels_after = visualizer.compute_bfs_levels(flow_dist_after, source)

        # initial residual (before)
        visualizer.visualize_graph(edges=flow_dist_before, output_path=os.path.join(iter_dir, "initial_residual.png"), title=f"Iteration {idx}: Residual Network (Before)", levels=levels_before, residual_edges=None, mode="residual")

        # augmenting path
        visualizer.visualize_graph(edges=flow_dist_before, output_path=os.path.join(iter_dir, "selected_augmented_path.png"), title=f"Iteration {idx}: Augmenting Path", levels=levels_before, highlight_path=path)

        # final residual (after)
        visualizer.visualize_graph(edges=flow_dist_after, output_path=os.path.join(iter_dir, "final_residual.png"), title=f"Iteration {idx}: Residual Network (After)", levels=levels_after, residual_edges=None, mode="residual")

    # Final flow graph
    if final_flow_dist:
        final_edges = [(u, v, f, c) for u, v, f, c in final_flow_dist]
    else:
        # Reconstruct final by applying all paths
        final_map = {}
        for u, v, c in initial_edges:
            final_map[(u, v)] = (0, c)
        for _, pth, fadded in path_history:
            apply_path_to_flow(final_map, pth, fadded)
        final_edges = flow_dict_to_edge_list(final_map, num_vertices)

    final_levels = visualizer.compute_bfs_levels(final_edges, source)
    visualizer.visualize_graph(edges=final_edges, output_path=os.path.join(graph_visuals_dir, "final_flow_graph.png"), title="Final Max Flow Network", levels=final_levels, mode="full")


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations from Ford–Fulkerson execution data")
    parser.add_argument("batch_folder", nargs="?", choices=["graphs", "graphs_v", "graphs_e"], help="Batch folder to visualize (optional)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")
    visuals_dir = os.path.join(project_root, "visuals")

    execution_files = find_all_execution_data(results_dir)
    if args.batch_folder:
        execution_files = [(gf, jp) for gf, jp in execution_files if gf == args.batch_folder]

    if not execution_files:
        print("No execution data files found. Run experiments first using batch_run or runner.")
        sys.exit(1)

    print(f"Found {len(execution_files)} execution data file(s)")
    for graph_folder, json_path in execution_files:
        try:
            data = load_graph_execution_data(json_path)
            graph_name = data["graph_name"]
            graph_folder_name = data.get("graph_folder", graph_folder)
            print(f"Processing: {graph_folder_name}/{graph_name}")
            generate_visuals_from_data(data, visuals_dir, graph_name, graph_folder_name)
            print(f"  ✓ Generated visuals for {graph_name}\n")
        except Exception as e:
            print(f"  ✗ Error processing {json_path}: {e}\n")

    print(f"✓ All visualizations generated in {visuals_dir}")


if __name__ == "__main__":
    main()