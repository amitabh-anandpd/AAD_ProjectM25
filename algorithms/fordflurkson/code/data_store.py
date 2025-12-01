"""
Data storage for Ford–Fulkerson execution data; mirrors Dinic's `data_store.py` layout
so the visualizer can read files without modification.
"""
import json
import os
from typing import List, Tuple, Dict, Any


def save_graph_execution_data(
    graph_name: str,
    num_vertices: int,
    source: int,
    sink: int,
    initial_edges: List[Tuple[int, int, int]],
    final_flow_dist: List[Tuple[int, int, int, int]],
    path_history: List[Tuple[int, List[int], int]],
    final_levels: List[int],
    results_dir: str,
    graph_folder: str = "graphs",
) -> str:
    os.makedirs(results_dir, exist_ok=True)
    organized_dir = os.path.join(results_dir, graph_folder)
    os.makedirs(organized_dir, exist_ok=True)

    data = {
        "graph_name": graph_name,
        "graph_folder": graph_folder,
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


def load_graph_execution_data(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r") as f:
        data = json.load(f)
    data["initial_edges"] = [tuple(e) for e in data.get("initial_edges", [])]
    data["final_flow_dist"] = [tuple(e) for e in data.get("final_flow_dist", [])]
    data["path_history"] = [(p[0], p[1], p[2]) for p in data.get("path_history", [])]
    if "graph_folder" not in data:
        data["graph_folder"] = "graphs"
    return data
