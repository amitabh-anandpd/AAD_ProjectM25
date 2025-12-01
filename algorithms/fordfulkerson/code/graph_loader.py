"""
Graph loading module for Dinic's algorithm.

This module provides functions to load graph files from disk.
In this module, I handle reading graph files in the standard format:
    first line: num_vertices num_edges
    subsequent lines: u v capacity (one edge per line)
"""
from typing import List, Tuple


def load_graph(graph_path: str) -> Tuple[int, int, List[Tuple[int, int, int]]]:
    """
    Load a graph from a text file.
    
    Args:
        graph_path: Path to the graph file
        
    Returns:
        Tuple of (num_vertices, num_edges, edges) where edges is a list of (u, v, capacity)
        
    Raises:
        ValueError: If the file format is invalid
        FileNotFoundError: If the file doesn't exist
    """
    edges: List[Tuple[int, int, int]] = []
    
    with open(graph_path, "r") as f:
        first = f.readline().strip().split()
        if len(first) < 2:
            raise ValueError("Invalid graph file: first line must contain 'vertices edges'")
        
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
    """
    Validate that source and sink are valid vertex indices.
    
    Args:
        num_vertices: Number of vertices in the graph
        source: Source vertex index
        sink: Sink vertex index
        
    Raises:
        ValueError: If source or sink is out of bounds
    """
    if not (0 <= source < num_vertices and 0 <= sink < num_vertices):
        raise ValueError(f"Source ({source}) or sink ({sink}) is out of bounds for graph with {num_vertices} vertices.")
    
    if source == sink:
        raise ValueError("Source and sink must be different vertices.")
