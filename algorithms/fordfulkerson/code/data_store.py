"""
Data storage module for saving graph execution data for later visualization.

This module handles saving and loading the data needed to regenerate visualizations
without re-running the algorithm. JSON files are organized by graph folder.
"""
import json
import os
from typing import List, Tuple, Dict, Any


def save_graph_execution_data(
    graph_name: str,
    num_vertices: int,
    source: int,
    sink: int,
    initial_edges: List[Tuple[int, int, int]],  # (u, v, capacity)
    final_flow_dist: List[Tuple[int, int, int, int]],  # (u, v, flow, capacity)
    path_history: List[Tuple[int, List[int], int]],  # (iteration, path, flow_added)
    final_levels: List[int],
    results_dir: str,
    graph_folder: str = "graphs",  # Folder name where graph came from (graphs, graphs2, etc.)
) -> str:
    """
    Save graph execution data to JSON file, organized by graph folder.
    
    Args:
        graph_name: Name of the graph (without extension)
        num_vertices: Number of vertices
        source: Source vertex index
        sink: Sink vertex index
        initial_edges: List of (u, v, capacity) for original edges
        final_flow_dist: List of (u, v, flow, capacity) for final state
        path_history: List of (iteration, path, flow_added) for each augmenting path
        final_levels: Final BFS level assignments
        results_dir: Directory to save the JSON file
        graph_folder: Name of the graph folder (graphs, graphs2, etc.)
        
    Returns:
        Path to the saved JSON file
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Create organized folder structure: results/graphs/, results/graphs2/, etc.
    organized_dir = os.path.join(results_dir, graph_folder)
    os.makedirs(organized_dir, exist_ok=True)
    
    data = {
        "graph_name": graph_name,
        "graph_folder": graph_folder,  # Store which folder it came from
        "num_vertices": num_vertices,
        "source": source,
        "sink": sink,
        "initial_edges": [[u, v, c] for u, v, c in initial_edges],
        "final_flow_dist": [[u, v, f, c] for u, v, f, c in final_flow_dist],
        "path_history": [[iter_num, path, flow] for iter_num, path, flow in path_history],
        "final_levels": final_levels,
    }
    
    json_path = os.path.join(organized_dir, f"{graph_name}_execution_data.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return json_path


def find_all_execution_data(results_dir: str) -> List[Tuple[str, str]]:
    """
    Find all execution data JSON files, organized by folder.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        List of (graph_folder, json_path) tuples
    """
    execution_files = []
    
    if not os.path.exists(results_dir):
        return execution_files
    
    # Look in organized folders (any folder that contains execution JSONs)
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            # Look for JSON files in this folder
            for filename in os.listdir(item_path):
                if filename.endswith("_execution_data.json"):
                    json_path = os.path.join(item_path, filename)
                    execution_files.append((item, json_path))
    
    # Also check root results_dir for legacy files
    for filename in os.listdir(results_dir):
        if filename.endswith("_execution_data.json"):
            json_path = os.path.join(results_dir, filename)
            execution_files.append(("graphs", json_path))  # Default to graphs folder
    
    return execution_files


def load_graph_execution_data(json_path: str) -> Dict[str, Any]:
    """
    Load graph execution data from JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Dictionary containing all execution data
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Convert back to tuples
    data["initial_edges"] = [tuple(e) for e in data["initial_edges"]]
    data["final_flow_dist"] = [tuple(e) for e in data["final_flow_dist"]]
    data["path_history"] = [(p[0], p[1], p[2]) for p in data["path_history"]]
    
    # Default graph_folder if not present (for legacy files)
    if "graph_folder" not in data:
        data["graph_folder"] = "graphs"
    
    return data
