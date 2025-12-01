"""
Push-Relabel Algorithm Implementation
Implements the maximum flow algorithm using the push-relabel method with FIFO vertex selection.
"""
import csv
import os
import shutil
import time
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


class Edge:
    """Represents a directed edge in the flow network."""
    def __init__(self, to: int, rev: int, capacity: int, is_original: bool):
        self.to = to
        self.rev = rev
        self.capacity = capacity
        self.flow = 0
        self.is_original = is_original

    @property
    def residual_capacity(self) -> int:
        """Returns the residual capacity of the edge."""
        return self.capacity - self.flow


class PushRelabel:
    """
    Push-Relabel Algorithm for Maximum Flow.
    
    Uses the FIFO (First-In-First-Out) heuristic for vertex selection,
    which provides good practical performance.
    """
    
    def __init__(self, graph_path: str, source: int, sink: int, logger: Optional[logging.Logger] = None):
        self.graph_path = graph_path
        self.source = source
        self.sink = sink
        self.num_vertices = 0
        self.num_edges_declared = 0
        self.graph: List[List[Edge]] = []
        self.original_edges: List[Tuple[int, int, int]] = []
        
        # Push-Relabel specific data structures
        self.height: List[int] = []
        self.excess: List[int] = []
        self.active_vertices: deque = deque()
        
        # Metrics tracking
        self.num_pushes = 0
        self.num_relabels = 0
        self.num_operations = 0
        self.push_time_total = 0.0
        self.relabel_time_total = 0.0
        
        # Visualization and output
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        self.visuals_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visuals")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.visuals_dir, exist_ok=True)
        
        self.logger = logger or logging.getLogger("push_relabel")
        
        graph_basename = os.path.splitext(os.path.basename(graph_path))[0]
        self.graph_visuals_dir = os.path.join(self.visuals_dir, graph_basename)
        os.makedirs(self.graph_visuals_dir, exist_ok=True)
        
        self._layout_positions: Optional[Dict[int, Tuple[float, float]]] = None
        self.snapshot_counter = 0
        
        self._read_graph()
        self._initialize_layout()

    def _read_graph(self) -> None:
        """Read graph from file."""
        with open(self.graph_path, "r") as f:
            first = f.readline().strip().split()
            if len(first) < 2:
                raise ValueError("Invalid graph file: first line must contain 'vertices edges'")
            self.num_vertices = int(first[0])
            self.num_edges_declared = int(first[1])
            self.graph = [[] for _ in range(self.num_vertices)]
            
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                u_str, v_str, c_str = line.split()
                u, v, c = int(u_str), int(v_str), int(c_str)
                self.add_edge(u, v, c)
                self.original_edges.append((u, v, c))
        
        if not (0 <= self.source < self.num_vertices and 0 <= self.sink < self.num_vertices):
            raise ValueError("Source or sink is out of bounds for the provided graph.")

    def add_edge(self, u: int, v: int, capacity: int) -> None:
        """Add edge with reverse residual edge."""
        forward = Edge(to=v, rev=len(self.graph[v]), capacity=capacity, is_original=True)
        backward = Edge(to=u, rev=len(self.graph[u]), capacity=0, is_original=False)
        self.graph[u].append(forward)
        backward.rev = len(self.graph[u]) - 1
        self.graph[v].append(backward)

    def _initialize_preflow(self) -> None:
        """Initialize preflow by saturating all edges from source."""
        self.height = [0] * self.num_vertices
        self.excess = [0] * self.num_vertices
        self.active_vertices.clear()
        
        # Set source height to n (number of vertices)
        self.height[self.source] = self.num_vertices
        
        # Saturate all edges from source
        for edge in self.graph[self.source]:
            if edge.capacity > 0:
                flow = edge.capacity
                edge.flow = flow
                self.graph[edge.to][edge.rev].flow = -flow
                self.excess[edge.to] += flow
                self.excess[self.source] -= flow
                
                # Add to active vertices if not sink
                if edge.to != self.sink and edge.to != self.source:
                    if edge.to not in self.active_vertices:
                        self.active_vertices.append(edge.to)

    def _push(self, u: int, edge: Edge) -> bool:
        """
        Push flow from vertex u through edge.
        Returns True if push was successful.
        """
        if self.excess[u] <= 0:
            return False
        if edge.residual_capacity <= 0:
            return False
        if self.height[u] != self.height[edge.to] + 1:
            return False
        
        # Calculate flow to push
        flow = min(self.excess[u], edge.residual_capacity)
        
        # Update flow
        edge.flow += flow
        self.graph[edge.to][edge.rev].flow -= flow
        
        # Update excess
        self.excess[u] -= flow
        self.excess[edge.to] += flow
        
        # Add to active vertices if became active
        if edge.to != self.sink and edge.to != self.source:
            if self.excess[edge.to] == flow:  # Just became active
                self.active_vertices.append(edge.to)
        
        self.num_pushes += 1
        return True

    def _relabel(self, u: int) -> None:
        """Relabel vertex u to the minimum valid height."""
        if self.excess[u] <= 0:
            return
        
        min_height = float('inf')
        for edge in self.graph[u]:
            if edge.residual_capacity > 0:
                min_height = min(min_height, self.height[edge.to])
        
        if min_height < float('inf'):
            self.height[u] = min_height + 1
            self.num_relabels += 1

    def _discharge(self, u: int) -> None:
        """Discharge excess flow from vertex u."""
        while self.excess[u] > 0:
            pushed = False
            
            # Try to push to all neighbors
            for edge in self.graph[u]:
                if self.excess[u] <= 0:
                    break
                if self._push(u, edge):
                    pushed = True
            
            # If no push was successful, relabel
            if not pushed and self.excess[u] > 0:
                self._relabel(u)

    def _initialize_layout(self) -> None:
        """Initialize layout for visualizations."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_vertices))
        for u, v, _ in self.original_edges:
            G.add_edge(u, v)
        
        try:
            self._layout_positions = nx.spring_layout(G, k=2, iterations=50)
        except:
            self._layout_positions = nx.kamada_kawai_layout(G)

    def _save_graph_visual(
        self,
        path: str,
        *,
        title: str,
        highlight_vertex: Optional[int] = None,
        mode: str = "full",
    ) -> None:
        """Save graph visualization with current flow and heights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pos = self._layout_positions or {}

        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.num_vertices))
        
        original_edges: List[Tuple[int, int, Dict]] = []
        for u in range(self.num_vertices):
            for edge in self.graph[u]:
                if edge.is_original:
                    data = {"flow": edge.flow, "capacity": edge.capacity}
                    original_edges.append((u, edge.to, data))
                    graph.add_edge(u, edge.to, **data)

        plt.figure(figsize=(14, 10))

        # Node colors based on role and excess
        node_colors = []
        node_labels = {}
        for node in graph.nodes:
            if node == self.source:
                node_colors.append("lightgreen")
            elif node == self.sink:
                node_colors.append("mediumpurple")
            elif highlight_vertex is not None and node == highlight_vertex:
                node_colors.append("gold")
            elif self.excess[node] > 0:
                node_colors.append("lightcoral")
            else:
                node_colors.append("white")
            
            # Label with height and excess
            node_labels[node] = f"{node}\nh={self.height[node]}\ne={self.excess[node]}"

        nx.draw_networkx_nodes(
            graph, pos, node_color=node_colors, edgecolors="black", 
            node_size=1500, linewidths=2
        )
        nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=9, font_weight="bold")

        # Draw edges with flow information
        if original_edges:
            for u, v, data in original_edges:
                flow = data["flow"]
                capacity = data["capacity"]
                
                # Color based on saturation
                if capacity > 0 and flow >= capacity:
                    color = "firebrick"
                    width = 2.5
                elif flow > 0:
                    color = "royalblue"
                    width = 2.0
                else:
                    color = "gray"
                    width = 1.5
                
                nx.draw_networkx_edges(
                    graph, pos, edgelist=[(u, v)],
                    edge_color=color, width=width,
                    alpha=0.8, arrows=True, arrowsize=20,
                    connectionstyle='arc3,rad=0.1',
                )

            # Edge labels
            edge_labels = {}
            for u, v, data in original_edges:
                edge_labels[(u, v)] = f'{data["flow"]}/{data["capacity"]}'
            
            nx.draw_networkx_edge_labels(
                graph, pos, edge_labels=edge_labels, font_size=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
            )

        # Legend
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                   markersize=12, label='Source'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='mediumpurple', 
                   markersize=12, label='Sink'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                   markersize=12, label='Active (excess > 0)'),
            Line2D([0], [0], color="gray", lw=2, label="Unused Edge"),
            Line2D([0], [0], color="royalblue", lw=2, label="Flowing Edge"),
            Line2D([0], [0], color="firebrick", lw=2, label="Saturated Edge"),
        ]
        if highlight_vertex is not None:
            legend_handles.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gold', 
                       markersize=12, label='Current Vertex')
            )
        
        plt.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9)
        plt.title(title, fontsize=13, pad=20)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

    def run(self) -> Tuple[int, float]:
        """Run Push-Relabel algorithm and return max flow and runtime."""
        self._reset_visual_outputs()
        self.num_pushes = 0
        self.num_relabels = 0
        self.num_operations = 0
        self.push_time_total = 0.0
        self.relabel_time_total = 0.0
        self.snapshot_counter = 0
        
        # Save initial graph
        initial_path = os.path.join(self.graph_visuals_dir, "initial_graph.png")
        self.height = [0] * self.num_vertices
        self.excess = [0] * self.num_vertices
        self._save_graph_visual(initial_path, title="Initial Flow Network")
        
        start = time.perf_counter()
        self.logger.info("Starting Push-Relabel algorithm: source=%s sink=%s", self.source, self.sink)
        
        # Initialize preflow
        init_start = time.perf_counter()
        self._initialize_preflow()
        init_time = time.perf_counter() - init_start
        self.logger.info("Preflow initialized in %.6fs", init_time)
        
        # Save after preflow
        preflow_path = os.path.join(self.graph_visuals_dir, "after_preflow.png")
        self._save_graph_visual(preflow_path, title="After Preflow Initialization")
        
        # Main loop: process active vertices
        operation_count = 0
        snapshot_interval = max(1, len(self.active_vertices) // 5)  # Take ~5 snapshots
        
        while self.active_vertices:
            u = self.active_vertices.popleft()
            
            if self.excess[u] <= 0:
                continue
            
            operation_count += 1
            self.num_operations += 1
            
            # Take periodic snapshots
            if operation_count % snapshot_interval == 0 or operation_count <= 3:
                self.snapshot_counter += 1
                snapshot_path = os.path.join(
                    self.graph_visuals_dir, 
                    f"operation_{self.snapshot_counter:03d}_vertex_{u}.png"
                )
                self._save_graph_visual(
                    snapshot_path,
                    title=f"Operation {self.snapshot_counter}: Processing Vertex {u}",
                    highlight_vertex=u
                )
            
            # Discharge vertex
            discharge_start = time.perf_counter()
            self._discharge(u)
            discharge_time = time.perf_counter() - discharge_start
            
            self.logger.debug("Discharged vertex %d in %.6fs", u, discharge_time)
        
        runtime = time.perf_counter() - start
        max_flow = sum(edge.flow for edge in self.graph[self.source] if edge.is_original)
        
        # Save final graph
        final_path = os.path.join(self.graph_visuals_dir, "final_flow_graph.png")
        self._save_graph_visual(final_path, title=f"Final Max Flow = {max_flow}")
        
        self.logger.info(
            "Algorithm complete. Max flow: %s in %.6fs (pushes=%d, relabels=%d)",
            max_flow, runtime, self.num_pushes, self.num_relabels
        )
        
        return max_flow, runtime

    def _reset_visual_outputs(self) -> None:
        """Clean up old visualizations."""
        if os.path.exists(self.graph_visuals_dir):
            shutil.rmtree(self.graph_visuals_dir, ignore_errors=True)
        os.makedirs(self.graph_visuals_dir, exist_ok=True)

    def compute_min_cut(self) -> Tuple[Set[int], Set[int], List[Tuple[int, int]]]:
        """
        Compute minimum cut from residual graph.
        Returns: (source_side, sink_side, cut_edges)
        """
        # BFS from source in residual graph
        visited = set()
        queue = deque([self.source])
        visited.add(self.source)
        
        while queue:
            u = queue.popleft()
            for edge in self.graph[u]:
                if edge.residual_capacity > 0 and edge.to not in visited:
                    visited.add(edge.to)
                    queue.append(edge.to)
        
        source_side = visited
        sink_side = set(range(self.num_vertices)) - visited
        
        # Find cut edges (from source_side to sink_side in original graph)
        cut_edges = []
        for u in source_side:
            for edge in self.graph[u]:
                if edge.is_original and edge.to in sink_side:
                    cut_edges.append((u, edge.to))
        
        return source_side, sink_side, cut_edges

    def _flow_distribution(self) -> List[Tuple[int, int, int, int]]:
        """Return list of (u, v, flow, capacity) for original edges."""
        dist: List[Tuple[int, int, int, int]] = []
        for u in range(self.num_vertices):
            for edge in self.graph[u]:
                if edge.is_original:
                    dist.append((u, edge.to, edge.flow, edge.capacity))
        
        order = {(u, v, c): i for i, (u, v, c) in enumerate(self.original_edges)}
        dist.sort(key=lambda x: order.get((x[0], x[1], x[3]), 0))
        return dist

    def write_outputs(self, max_flow: int, runtime: float) -> None:
        """Write output.txt with flow distribution and min-cut."""
        output_path = os.path.join(self.results_dir, "output.txt")
        lines = []
        lines.append(f"Maximum Flow: {max_flow}")
        lines.append("")
        
        # Flow distribution
        lines.append("Flow Distribution:")
        for u, v, flow, cap in self._flow_distribution():
            lines.append(f"{u} → {v} : {flow} / {cap}")
        lines.append("")
        
        # Minimum cut
        source_side, sink_side, cut_edges = self.compute_min_cut()
        lines.append("Minimum Cut:")
        lines.append(f"Source Side (S): {sorted(source_side)}")
        lines.append(f"Sink Side (T): {sorted(sink_side)}")
        lines.append(f"Cut Edges: {cut_edges}")
        
        cut_capacity = sum(
            edge.capacity for u in range(self.num_vertices) 
            for edge in self.graph[u] 
            if edge.is_original and u in source_side and edge.to in sink_side
        )
        lines.append(f"Cut Capacity: {cut_capacity}")
        lines.append("")
        lines.append(f"Max-Flow Min-Cut Theorem Verified: {max_flow == cut_capacity}")
        lines.append("")
        
        # Algorithm metrics
        lines.append(f"Runtime: {runtime:.6f}s")
        lines.append(f"Number of Push Operations: {self.num_pushes}")
        lines.append(f"Number of Relabel Operations: {self.num_relabels}")
        lines.append(f"Total Operations: {self.num_operations}")
        
        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def append_metrics(
        self, graph_name: str, max_flow: int, runtime: float, 
        family: str = "", seed: int = 0, trial: int = 0
    ) -> None:
        """Append summary metrics to performance.csv."""
        summary_path = os.path.join(self.results_dir, "performance.csv")
        header = [
            "graph_filename", "family", "n", "m", "seed", "trial", "max_flow", 
            "total_time", "num_pushes", "num_relabels", "num_operations"
        ]
        
        file_exists = os.path.exists(summary_path)
        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow([
                graph_name, family, self.num_vertices, self.num_edges_declared,
                seed, trial, max_flow, f"{runtime:.6f}",
                self.num_pushes, self.num_relabels, self.num_operations
            ])
