"""
Graph visualization module (copied from Dinic's visualizer).

Provides `GraphVisualizer` used by `visualizer_script.py` to render
flow networks, residual graphs, and augmenting paths.
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
    def __init__(self, num_vertices: int, source: int, sink: int):
        self.num_vertices = num_vertices
        self.source = source
        self.sink = sink
        self._layout_positions: Optional[Dict[int, Tuple[float, float]]] = None
        self._level_map: List[int] = []

    def compute_bfs_levels(self, edges: List[Tuple[int, int, int, int]], source: int) -> List[int]:
        adj: DefaultDict[int, List[int]] = defaultdict(list)
        for u, v, flow, capacity in edges:
            if capacity > 0:
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
        level_map: DefaultDict[int, List[int]] = defaultdict(list)
        for node, level_val in enumerate(levels):
            if level_val >= 0:
                level_map[level_val].append(node)
            else:
                level_map[-1].append(node)

        pos: Dict[int, Tuple[float, float]] = {}
        max_level = max([l for l in level_map.keys() if l >= 0], default=0)
        unreachable_level = max_level + 1 if -1 in level_map else max_level
        max_nodes_in_level = max(len(nodes) for nodes in level_map.values()) if level_map else 1
        vertical_spacing = max(1.5, max_nodes_in_level * 0.6)

        for level_idx in sorted(level_map.keys()):
            nodes = sorted(level_map[level_idx])
            if not nodes:
                continue
            x = float(level_idx) if level_idx >= 0 else float(unreachable_level)
            if len(nodes) == 1:
                pos[nodes[0]] = (x, 0.0)
            else:
                y_start = -vertical_spacing / 2
                y_step = vertical_spacing / (len(nodes) - 1)
                for i, node in enumerate(nodes):
                    pos[node] = (x, y_start + i * y_step)

        return pos

    def visualize_graph(self, edges: List[Tuple[int, int, int, int]], output_path: str, title: str,
                        levels: Optional[List[int]] = None, highlight_path: Optional[List[int]] = None,
                        residual_edges: Optional[List[Tuple[int, int, int, bool]]] = None, mode: str = "full") -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if levels is None:
            levels = self.compute_bfs_levels(edges, self.source)
        self._level_map = levels[:]
        pos = self._compute_level_based_layout(levels)

        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.num_vertices))

        original_edge_data: List[Tuple[int, int, int, int]] = []
        for u, v, flow, capacity in edges:
            original_edge_data.append((u, v, flow, capacity))
            graph.add_edge(u, v, flow=flow, capacity=capacity)

        plt.figure(figsize=(14, 10))
        dpi = 250

        node_colors = []
        for node in sorted(graph.nodes()):
            if node == self.source:
                node_colors.append("green")
            elif node == self.sink:
                node_colors.append("purple")
            else:
                node_colors.append("lightblue")

        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, edgecolors="black", node_size=1200, linewidths=2.0)
        nx.draw_networkx_labels(graph, pos, font_size=12, font_weight="bold")

        path_edges = set()
        if highlight_path:
            for i in range(len(highlight_path) - 1):
                path_edges.add((highlight_path[i], highlight_path[i + 1]))

        sorted_edges = sorted(original_edge_data, key=lambda e: (e[0], e[1]))

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

        if unused_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=[(u, v) for u, v, _, _ in unused_edges], edge_color="#B0B0B0", width=1.5, style="solid", alpha=1.0, arrows=True, arrowsize=25, connectionstyle="arc3,rad=0.1")

        if flowing_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=[(u, v) for u, v, _, _ in flowing_edges], edge_color="#1C75BC", width=2.5, style="solid", alpha=1.0, arrows=True, arrowsize=25, connectionstyle="arc3,rad=0.1")

        if saturated_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=[(u, v) for u, v, _, _ in saturated_edges], edge_color="#D62728", width=3.0, style="solid", alpha=1.0, arrows=True, arrowsize=25, connectionstyle="arc3,rad=0.1")

        if augmenting_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=[(u, v) for u, v, _, _ in augmenting_edges], edge_color="#00AEEF", width=3.0, style="solid", alpha=1.0, arrows=True, arrowsize=25, connectionstyle="arc3,rad=0.1")

        edge_labels = {}
        for u, v, flow, capacity in sorted_edges:
            edge_labels[(u, v)] = f"{flow}/{capacity}"

        if edge_labels:
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10, font_weight="bold", bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

        if mode == "residual" and residual_edges:
            residual_forward = [(u, v, cap) for u, v, cap, is_orig in residual_edges if is_orig and cap > 0]
            residual_backward = [(u, v, cap) for u, v, cap, is_orig in residual_edges if not is_orig and cap > 0]
            if residual_forward:
                nx.draw_networkx_edges(graph, pos, edgelist=[(u, v) for u, v, _ in residual_forward], edge_color="orange", width=2.0, style="dashed", alpha=0.6, arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")
            if residual_backward:
                nx.draw_networkx_edges(graph, pos, edgelist=[(u, v) for u, v, _ in residual_backward], edge_color="#FFA500", width=2.0, style="dashed", alpha=0.4, arrows=True, arrowsize=20, connectionstyle="arc3,rad=0.1")

        for node in sorted(graph.nodes()):
            if node in pos:
                x, y = pos[node]
                level_val = levels[node] if node < len(levels) else -1
                if level_val >= 0:
                    plt.text(x, y + 0.25, f"L{level_val}", fontsize=9, color='lightgray', ha='center', va='bottom')

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
            legend_handles.append(Line2D([0], [0], color="#00AEEF", lw=3.0, label="Augmenting Path"))

        plt.legend(handles=legend_handles, loc="upper left", fontsize=10, framealpha=0.95)
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()

    def generate_histograms(self, edges: List[Tuple[int, int, int, int]], path_history: List[Tuple[int, List[int], int]], output_dir: str, graph_name: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        path_lengths = [len(path) - 1 for _, path, _ in path_history]
        if path_lengths:
            plt.figure(figsize=(8, 6))
            plt.hist(path_lengths, bins=range(min(path_lengths), max(path_lengths) + 2), edgecolor='black', alpha=0.7, color='#1C75BC')
            plt.xlabel("Augmenting Path Length (number of edges)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.title(f"{graph_name}: Distribution of Augmenting Path Lengths", fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "path_lengths_histogram.png"), dpi=200, bbox_inches="tight")
            plt.close()

        utilizations = []
        for u, v, flow, capacity in edges:
            if capacity > 0:
                utilizations.append(flow / capacity)

        if utilizations:
            plt.figure(figsize=(8, 6))
            bins = np.linspace(0, 1.0, 21)
            plt.hist(utilizations, bins=bins, edgecolor='black', alpha=0.7, color='#D62728')
            plt.xlabel("Edge Utilization (flow / capacity)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.title(f"{graph_name}: Distribution of Edge Utilizations", fontsize=14, fontweight='bold')
            plt.xlim(0, 1.0)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "edge_utilization_histogram.png"), dpi=200, bbox_inches="tight")
            plt.close()

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