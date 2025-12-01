"""
Graph loader for Ford–Fulkerson - copied to mirror Dinic's behavior exactly.
"""
from typing import List, Tuple


def load_graph(graph_path: str) -> Tuple[int, int, List[Tuple[int, int, int]]]:
    edges: List[Tuple[int, int, int]] = []
    with open(graph_path, "r") as f:
        first = f.readline().strip().split()
        num_vertices = int(first[0])
        num_edges_declared = int(first[1])
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            u, v, c = int(parts[0]), int(parts[1]), int(parts[2])
            edges.append((u, v, c))
    return num_vertices, num_edges_declared, edges


def validate_graph(num_vertices: int, source: int, sink: int) -> None:
    if not (0 <= source < num_vertices and 0 <= sink < num_vertices):
        raise ValueError("Source or sink out of bounds")
    if source == sink:
        raise ValueError("Source and sink must be different")
