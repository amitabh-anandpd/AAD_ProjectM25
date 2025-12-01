"""
Graph visualization module for Dinic's algorithm.

This module provides functions to visualize flow networks, residual graphs,
and augmenting paths with deterministic level-based layouts.
"""
import os
from collections import deque, defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import numpy as np


class GraphVisualizer:
    """
    Visualizer for flow networks and residual graphs.
    
    This class implements a deterministic level-based layout where:
    - X-coordinate = BFS level (source at left, sink at right)
    - Y-coordinate = equally spaced within each level
    - Same graph always produces same layout
    """
    
    def __init__(self, num_vertices: int, source: int, sink: int):
        """
        Initialize visualizer.
        
        Args:
            num_vertices: Number of vertices in the graph
            source: Source vertex index
            sink: Sink vertex index
        """
        self.num_vertices = num_vertices
        self.source = source
        self.sink = sink
        self._layout_positions: Optional[Dict[int, Tuple[float, float]]] = None
        self._level_map: List[int] = []
    
    def compute_bfs_levels(self, edges: List[Tuple[int, int, int, int]], 
                          source: int) -> List[int]:
        """
        Compute BFS levels from source using current flow graph.
        
        Here I run BFS to determine the shortest path distance (level)
        from source to each node, considering only edges with positive capacity.
        
        Args:
            edges: List of (u, v, flow, capacity) for original edges
            source: Source vertex
            
        Returns:
            List of level assignments for each vertex (-1 if unreachable)
        """
        # Build adjacency list
        adj: DefaultDict[int, List[int]] = defaultdict(list)
        for u, v, flow, capacity in edges:
            if capacity > 0:  # Only consider edges with capacity
                adj[u].append(v)
        
        levels = [-1] * self.num_vertices
        levels[source] = 0
        queue: deque[int] = deque([source])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if levels[v] == -1:
                    levels[v] = levels[u] + 1
                    queue.append(v)
        
        return levels
    
    def _compute_level_based_layout(self, levels: List[int]) -> Dict[int, Tuple[float, float]]:
        """
        Compute deterministic level-based layout.
        
        In this function, I create a layout where:
        - X-coordinate = BFS level (source at x=0, sink at rightmost)
        - Y-coordinate = equally spaced within each level (sorted by node ID for determinism)
        - No randomness, same graph always produces same layout
        
        Args:
            levels: List of level assignments for each vertex
            
        Returns:
            Dictionary mapping node -> (x, y) position
        """
        # Group nodes by level
        level_map: DefaultDict[int, List[int]] = defaultdict(list)
        for node, level_val in enumerate(levels):
            if level_val >= 0:
                level_map[level_val].append(node)
            else:
                level_map[-1].append(node)  # Unreachable nodes
        
        pos: Dict[int, Tuple[float, float]] = {}
        
        # Find maximum level
        max_level = max([l for l in level_map.keys() if l >= 0], default=0)
        unreachable_level = max_level + 1 if -1 in level_map else max_level
        
        # Calculate vertical spacing based on max nodes in any level
        max_nodes_in_level = max(len(nodes) for nodes in level_map.values()) if level_map else 1
        vertical_spacing = max(1.5, max_nodes_in_level * 0.6)  # Minimum 1.5 units
        
        # Position nodes level by level
        for level_idx in sorted(level_map.keys()):
            nodes = sorted(level_map[level_idx])  # Sort for determinism
            if not nodes:
                continue
            
            # X-coordinate = level
            if level_idx >= 0:
                x = float(level_idx)
            else:
                x = float(unreachable_level)
            
            # Y-coordinate = equally spaced
            if len(nodes) == 1:
                pos[nodes[0]] = (x, 0.0)
            else:
                y_start = -vertical_spacing / 2
                y_step = vertical_spacing / (len(nodes) - 1)
                for i, node in enumerate(nodes):
                    pos[node] = (x, y_start + i * y_step)
        
        return pos
    
    def visualize_graph(
        self,
        edges: List[Tuple[int, int, int, int]],  # (u, v, flow, capacity)
        output_path: str,
        title: str,
        levels: Optional[List[int]] = None,
        highlight_path: Optional[List[int]] = None,
        residual_edges: Optional[List[Tuple[int, int, int, bool]]] = None,
        mode: str = "full",
    ) -> None:
        """
        Visualize flow network with deterministic level-based layout.
        
        Args:
            edges: List of (u, v, flow, capacity) for original edges
            output_path: Path to save the image
            title: Title for the plot
            levels: Optional level assignments (computed from source if None)
            highlight_path: Optional list of nodes to highlight as augmenting path
            residual_edges: Optional list of (u, v, residual_capacity, is_original) for residual graph
            mode: "full" for flow graph, "residual" for residual graph
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Compute or use provided levels
        if levels is None:
            levels = self.compute_bfs_levels(edges, self.source)
        
        self._level_map = levels[:]
        
        # Compute deterministic layout
        pos = self._compute_level_based_layout(levels)
        self._layout_positions = pos
        
        # Create graph structure
        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.num_vertices))
        
        # Add original edges
        original_edge_data: List[Tuple[int, int, int, int]] = []
        for u, v, flow, capacity in edges:
            original_edge_data.append((u, v, flow, capacity))
            graph.add_edge(u, v, flow=flow, capacity=capacity)
        
        # Create figure
        plt.figure(figsize=(14, 10))
        dpi = 250
        
        # Draw nodes with specified colors
        node_colors = []
        for node in sorted(graph.nodes()):  # Sort for determinism
            if node == self.source:
                node_colors.append("green")  # Source = green
            elif node == self.sink:
                node_colors.append("purple")  # Sink = purple
            else:
                node_colors.append("lightblue")  # Others = light blue
        
        nx.draw_networkx_nodes(
            graph, pos, 
            node_color=node_colors,
            edgecolors="black",
            node_size=1200,
            linewidths=2.0
        )
        nx.draw_networkx_labels(graph, pos, font_size=12, font_weight="bold")
        
        # Build edge path for highlighting check
        path_edges = set()
        if highlight_path:
            for i in range(len(highlight_path) - 1):
                path_edges.add((highlight_path[i], highlight_path[i + 1]))
        
        # Draw edges with specified styles
        # Sort edges for determinism: by (u, v) tuple
        sorted_edges = sorted(original_edge_data, key=lambda e: (e[0], e[1]))
        
        # Separate edges by type for z-ordering
        augmenting_edges = []
        saturated_edges = []
        flowing_edges = []
        unused_edges = []
        
        for u, v, flow, capacity in sorted_edges:
            if (u, v) in path_edges:
                augmenting_edges.append((u, v, flow, capacity))
            elif capacity > 0 and flow >= capacity:
                saturated_edges.append((u, v, flow, capacity))
            elif flow > 0:
                flowing_edges.append((u, v, flow, capacity))
            else:
                unused_edges.append((u, v, flow, capacity))
        
        # Draw unused edges (lowest z-order)
        if unused_edges:
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=[(u, v) for u, v, _, _ in unused_edges],
                edge_color="#B0B0B0",  # Gray
                width=1.5,
                style="solid",
                alpha=1.0,
                arrows=True,
                arrowsize=25,
                connectionstyle="arc3,rad=0.1",
            )
        
        # Draw flowing edges
        if flowing_edges:
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=[(u, v) for u, v, _, _ in flowing_edges],
                edge_color="#1C75BC",  # Blue
                width=2.5,
                style="solid",
                alpha=1.0,
                arrows=True,
                arrowsize=25,
                connectionstyle="arc3,rad=0.1",
            )
        
        # Draw saturated edges
        if saturated_edges:
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=[(u, v) for u, v, _, _ in saturated_edges],
                edge_color="#D62728",  # Red
                width=3.0,
                style="solid",
                alpha=1.0,
                arrows=True,
                arrowsize=25,
                connectionstyle="arc3,rad=0.1",
            )
        
        # Draw augmenting path edges (highest z-order)
        if augmenting_edges:
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=[(u, v) for u, v, _, _ in augmenting_edges],
                edge_color="#00AEEF",  # Cyan
                width=3.0,
                style="solid",
                alpha=1.0,
                arrows=True,
                arrowsize=25,
                connectionstyle="arc3,rad=0.1",
            )
        
        # Draw edge labels: flow/capacity
        edge_labels = {}
        for u, v, flow, capacity in sorted_edges:
            edge_labels[(u, v)] = f"{flow}/{capacity}"
        
        if edge_labels:
            nx.draw_networkx_edge_labels(
                graph, pos,
                edge_labels=edge_labels,
                font_size=10,
                font_weight="bold",
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2),
            )
        
        # Draw residual edges if in residual mode
        if mode == "residual" and residual_edges:
            residual_forward = [(u, v, cap) for u, v, cap, is_orig in residual_edges if is_orig and cap > 0]
            residual_backward = [(u, v, cap) for u, v, cap, is_orig in residual_edges if not is_orig and cap > 0]
            
            # Draw forward residual edges (dashed orange)
            if residual_forward:
                nx.draw_networkx_edges(
                    graph, pos,
                    edgelist=[(u, v) for u, v, _ in residual_forward],
                    edge_color="orange",
                    width=2.0,
                    style="dashed",
                    alpha=0.6,
                    arrows=True,
                    arrowsize=20,
                    connectionstyle="arc3,rad=0.1",
                )
            
            # Draw backward residual edges (dashed light-orange)
            if residual_backward:
                nx.draw_networkx_edges(
                    graph, pos,
                    edgelist=[(u, v) for u, v, _ in residual_backward],
                    edge_color="#FFA500",  # Light orange
                    width=2.0,
                    style="dashed",
                    alpha=0.4,
                    arrows=True,
                    arrowsize=20,
                    connectionstyle="arc3,rad=0.1",
                )
        
        # Draw level labels above nodes
        for node in sorted(graph.nodes()):
            if node in pos:
                x, y = pos[node]
                level_val = levels[node] if node < len(levels) else -1
                if level_val >= 0:
                    plt.text(
                        x, y + 0.25,
                        f"L{level_val}",
                        fontsize=9,
                        color='lightgray',
                        ha='center',
                        va='bottom'
                    )
        
        # Create legend
        legend_handles = [
            Line2D([0], [0], color="#B0B0B0", lw=1.5, label="Unused Edge"),
            Line2D([0], [0], color="#1C75BC", lw=2.5, label="Flowing Edge"),
            Line2D([0], [0], color="#D62728", lw=3.0, label="Saturated Edge"),
        ]
        
        if mode == "residual":
            legend_handles.extend([
                Line2D([0], [0], color="orange", lw=2, linestyle="--", alpha=0.6, label="Residual Forward"),
                Line2D([0], [0], color="#FFA500", lw=2, linestyle="--", alpha=0.4, label="Residual Backward"),
            ])
        
        if highlight_path:
            legend_handles.append(
                Line2D([0], [0], color="#00AEEF", lw=3.0, label="Augmenting Path")
            )
        
        plt.legend(handles=legend_handles, loc="upper left", fontsize=10, framealpha=0.95)
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    
    def generate_histograms(
        self,
        edges: List[Tuple[int, int, int, int]],
        path_history: List[Tuple[int, List[int], int]],  # (iteration, path, flow_added)
        output_dir: str,
        graph_name: str,
    ) -> None:
        """
        Generate histograms for demonstration graphs.
        
        Creates histograms of:
        - Augmenting path lengths
        - Residual edge utilizations
        - BFS levels distribution
        
        Args:
            edges: List of (u, v, flow, capacity) for original edges
            path_history: List of augmenting paths found
            output_dir: Directory to save histogram images
            graph_name: Name of the graph
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Histogram of augmenting path lengths
        path_lengths = [len(path) - 1 for _, path, _ in path_history]  # Length = edges, not nodes
        
        if path_lengths:
            plt.figure(figsize=(8, 6))
            plt.hist(path_lengths, bins=range(min(path_lengths), max(path_lengths) + 2), 
                    edgecolor='black', alpha=0.7, color='#1C75BC')
            plt.xlabel("Augmenting Path Length (number of edges)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.title(f"{graph_name}: Distribution of Augmenting Path Lengths", fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "path_lengths_histogram.png"), dpi=200, bbox_inches="tight")
            plt.close()
        
        # 2. Histogram of residual edge utilizations
        # Compute utilization = flow / capacity for each edge
        utilizations = []
        for u, v, flow, capacity in edges:
            if capacity > 0:
                utilization = flow / capacity
                utilizations.append(utilization)
        
        if utilizations:
            plt.figure(figsize=(8, 6))
            bins = np.linspace(0, 1.0, 21)  # 20 bins from 0 to 1
            plt.hist(utilizations, bins=bins, edgecolor='black', alpha=0.7, color='#D62728')
            plt.xlabel("Edge Utilization (flow / capacity)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.title(f"{graph_name}: Distribution of Edge Utilizations", fontsize=14, fontweight='bold')
            plt.xlim(0, 1.0)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "edge_utilization_histogram.png"), dpi=200, bbox_inches="tight")
            plt.close()
        
        # 3. Histogram of BFS levels distribution
        levels = self.compute_bfs_levels(edges, self.source)
        level_counts = defaultdict(int)
        for level in levels:
            if level >= 0:
                level_counts[level] += 1
        
        if level_counts:
            level_values = sorted(level_counts.keys())
            counts = [level_counts[l] for l in level_values]
            
            plt.figure(figsize=(8, 6))
            plt.bar(level_values, counts, edgecolor='black', alpha=0.7, color='#00AEEF')
            plt.xlabel("BFS Level", fontsize=12)
            plt.ylabel("Number of Nodes", fontsize=12)
            plt.title(f"{graph_name}: Distribution of Nodes by BFS Level", fontsize=14, fontweight='bold')
            plt.xticks(level_values)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "bfs_levels_histogram.png"), dpi=200, bbox_inches="tight")
            plt.close()
