import csv
import os
import shutil
import time
import logging
from collections import deque, defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple
import numpy as np

import matplotlib

# Use a non-interactive backend suitable for servers/CLI
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


class Edge:
    def __init__(self, to: int, rev: int, capacity: int, is_original: bool):
        self.to = to
        self.rev = rev
        self.capacity = capacity
        self.flow = 0
        self.is_original = is_original

    @property
    def residual_capacity(self) -> int:
        return self.capacity - self.flow


class Dinics:
    def __init__(self, graph_path: str, source: int, sink: int, logger: Optional[logging.Logger] = None):
        self.graph_path = graph_path
        self.source = source
        self.sink = sink
        self.num_vertices = 0
        self.num_edges_declared = 0
        self.level: List[int] = []
        self.it_ptr: List[int] = []
        self.graph: List[List[Edge]] = []
        self.original_edges: List[Tuple[int, int, int]] = []  # (u, v, capacity) for reporting/visualization
        self.iteration_index = 0
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        self.visuals_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "visuals")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.visuals_dir, exist_ok=True)
        self.logger = logger or logging.getLogger("dinics")

        self.iteration_metrics: List[Dict[str, object]] = []
        self._layout_positions: Optional[Dict[int, Tuple[float, float]]] = None
        self.total_augmenting_paths = 0
        self.bfs_phases = 0
        self.bfs_time_total = 0.0
        self.dfs_time_total = 0.0

        # Create graph-specific visuals folder
        graph_basename = os.path.splitext(os.path.basename(graph_path))[0]
        self.graph_visuals_dir = os.path.join(self.visuals_dir, graph_basename)
        os.makedirs(self.graph_visuals_dir, exist_ok=True)

        self._read_graph()
        self._initialize_layout()

    def _read_graph(self) -> None:
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
        # Basic validation
        if not (0 <= self.source < self.num_vertices and 0 <= self.sink < self.num_vertices):
            raise ValueError("Source or sink is out of bounds for the provided graph.")

    def _get_level_based_layout(self) -> Dict[int, Tuple[float, float]]:
        """Create level-based layout: x = level, y = jitter within level."""
        if not self.level or len(self.level) != self.num_vertices:
            self._get_level_map()  # Initialize levels
        
        level_map: DefaultDict[int, List[int]] = defaultdict(list)
        for node, level_val in enumerate(self.level):
            if level_val >= 0:
                level_map[level_val].append(node)
            else:
                level_map[-1].append(node)  # Unreachable
        
        pos: Dict[int, Tuple[float, float]] = {}
        max_level = max(level_map.keys()) if level_map else 0
        
        for level_idx in sorted(level_map.keys()):
            nodes = level_map[level_idx]
            if not nodes:
                continue
            x = level_idx if level_idx >= 0 else max_level + 1
            # Distribute nodes vertically within level
            if len(nodes) == 1:
                pos[nodes[0]] = (x, 0.0)
            else:
                y_spacing = 2.0 / (len(nodes) - 1) if len(nodes) > 1 else 0
                y_start = -1.0
                for i, node in enumerate(nodes):
                    pos[node] = (x, y_start + i * y_spacing)
        
        # Refine with kamada_kawai for better spacing within each level
        try:
            G = nx.DiGraph()
            G.add_nodes_from(range(self.num_vertices))
            for u, v, _ in self.original_edges:
                G.add_edge(u, v)
            refined = nx.kamada_kawai_layout(G)
            # Blend: use level-based x, kamada_kawai y
            for node in pos:
                if node in refined:
                    pos[node] = (pos[node][0], refined[node][1] * 0.5)
        except:
            pass  # Fallback to pure level-based
        
        return pos

    def _initialize_layout(self) -> None:
        """Initialize layout using level-based approach."""
        self._layout_positions = self._get_level_based_layout()

    def _reset_visual_outputs(self) -> None:
        """Clean up old visualizations for this graph."""
        if os.path.exists(self.graph_visuals_dir):
            shutil.rmtree(self.graph_visuals_dir, ignore_errors=True)
        os.makedirs(self.graph_visuals_dir, exist_ok=True)

    def add_edge(self, u: int, v: int, capacity: int) -> None:
        """Add edge with reverse residual edge. Mark forward as original."""
        forward = Edge(to=v, rev=len(self.graph[v]), capacity=capacity, is_original=True)
        backward = Edge(to=u, rev=len(self.graph[u]), capacity=0, is_original=False)
        self.graph[u].append(forward)
        backward.rev = len(self.graph[u]) - 1
        self.graph[v].append(backward)

    def _bfs_level_graph(self) -> bool:
        """Build level graph using BFS. Returns True if sink is reachable."""
        self.level = [-1] * self.num_vertices
        queue: deque[int] = deque()
        self.level[self.source] = 0
        queue.append(self.source)
        while queue:
            u = queue.popleft()
            for e in self.graph[u]:
                if e.residual_capacity > 0 and self.level[e.to] < 0:
                    self.level[e.to] = self.level[u] + 1
                    queue.append(e.to)
        return self.level[self.sink] >= 0

    def _dfs_find_path(self, u: int, pushed: int, path: List[int]) -> Tuple[int, List[int]]:
        """Find a single augmenting path. Returns (flow, path_nodes)."""
        if u == self.sink:
            return pushed, path.copy()
        while self.it_ptr[u] < len(self.graph[u]):
            e = self.graph[u][self.it_ptr[u]]
            if e.residual_capacity > 0 and self.level[e.to] == self.level[u] + 1:
                path.append(e.to)
                flow, found_path = self._dfs_find_path(e.to, min(pushed, e.residual_capacity), path)
                if flow > 0:
                    path.pop()
                    return flow, found_path
                path.pop()
            self.it_ptr[u] += 1
        return 0, []

    def _apply_path(self, path: List[int], flow: int) -> None:
        """Apply flow along the given path."""
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            for e in self.graph[u]:
                if e.to == v:
                    e.flow += flow
                    self.graph[v][e.rev].flow -= flow
                    break

    def _get_level_map(self) -> DefaultDict[int, List[int]]:
        """Get current level mapping for visualization."""
        level_map: DefaultDict[int, List[int]] = defaultdict(list)
        if not self.level or len(self.level) != self.num_vertices:
            # Initialize levels if not set
            self.level = [-1] * self.num_vertices
            self.level[self.source] = 0
            # Do a quick BFS to get initial levels
            queue: deque[int] = deque([self.source])
            while queue:
                u = queue.popleft()
                for e in self.graph[u]:
                    if e.capacity > 0 and self.level[e.to] < 0:
                        self.level[e.to] = self.level[u] + 1
                        queue.append(e.to)
        for node, level_val in enumerate(self.level):
            if level_val >= 0:
                level_map[level_val].append(node)
        # Include unreachable nodes
        for node in range(self.num_vertices):
            if node < len(self.level) and self.level[node] < 0:
                level_map[-1].append(node)  # Unreachable
        return level_map

    def _snapshot_flows(self) -> Dict[Tuple[int, int], int]:
        """Snapshot current flow values for comparison."""
        snapshot: Dict[Tuple[int, int], int] = {}
        for u in range(self.num_vertices):
            for e in self.graph[u]:
                if e.is_original:
                    snapshot[(u, e.to)] = e.flow
        return snapshot

    def _save_graph_visual(
        self,
        path: str,
        *,
        title: str,
        highlight_path: Optional[List[int]] = None,
        mode: str = "full",
    ) -> None:
        """Save graph visualization with improved styling."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Update layout based on current levels
        self._layout_positions = self._get_level_based_layout()
        pos = self._layout_positions or {}

        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.num_vertices))
        original_edges: List[Tuple[int, int, Dict[str, int]]] = []
        for u in range(self.num_vertices):
            for e in self.graph[u]:
                if e.is_original:
                    data = {"flow": e.flow, "capacity": e.capacity}
                    original_edges.append((u, e.to, data))
                    graph.add_edge(u, e.to, **data)

        residual_forward: List[Tuple[int, int, Dict[str, int]]] = []
        residual_reverse: List[Tuple[int, int, Dict[str, int]]] = []
        for u in range(self.num_vertices):
            for e in self.graph[u]:
                residual = e.residual_capacity
                if residual <= 0:
                    continue
                info = {"capacity": residual}
                if e.is_original:
                    residual_forward.append((u, e.to, info))
                else:
                    residual_reverse.append((u, e.to, info))

        plt.figure(figsize=(12, 8))

        # Node colors: lightgreen for source, mediumpurple for sink, white for others
        node_colors = []
        for node in graph.nodes:
            if node == self.source:
                node_colors.append("lightgreen")
            elif node == self.sink:
                node_colors.append("mediumpurple")
            else:
                node_colors.append("white")
        nx.draw_networkx_nodes(
            graph, pos, node_color=node_colors, edgecolors="black", node_size=1000, linewidths=2
        )
        nx.draw_networkx_labels(graph, pos, font_size=11, font_weight="bold")

        # Draw original edges with improved styling
        draw_edges = [(u, v) for u, v, _ in original_edges]
        if draw_edges:
            colors: List[str] = []
            widths: List[float] = []
            styles: List[str] = []
            zorders: List[int] = []
            for u, v, data in original_edges:
                edge = (u, v)
                # Check if edge is in highlighted path
                is_in_path = False
                if highlight_path:
                    for i in range(len(highlight_path) - 1):
                        if highlight_path[i] == u and highlight_path[i + 1] == v:
                            is_in_path = True
                            break

                if is_in_path:
                    color = "deepskyblue"  # Cyan for augmenting path
                    width = 3.5
                    style = "solid"
                    zorder = 5
                elif data["capacity"] > 0 and data["flow"] >= data["capacity"]:
                    color = "firebrick"  # Red for saturated
                    width = 2.5
                    style = "solid"
                    zorder = 2
                elif data["flow"] > 0:
                    color = "royalblue"  # Blue for flowing
                    width = 2.0
                    style = "solid"
                    zorder = 2
                else:
                    color = "gray"  # Gray for unused
                    width = 1.5
                    style = "solid"
                    zorder = 1
                colors.append(color)
                widths.append(width)
                styles.append(style)
                zorders.append(zorder)

            alpha = 0.95 if mode != "residual" else 0.4
            # Draw edges with curved style to avoid overlap, sorted by zorder
            edge_data = list(zip(draw_edges, colors, widths, styles, zorders))
            edge_data.sort(key=lambda x: x[4])  # Sort by zorder
            
            for (u, v), color, width, style, _ in edge_data:
                nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=[(u, v)],
                    edge_color=color,
                    width=width,
                    style=style,
                    alpha=alpha,
                    arrows=True,
                    arrowsize=25,
                    connectionstyle='arc3,rad=0.08',
                )

            # Edge labels with white background
            edge_labels = {}
            for u, v, data in original_edges:
                edge = (u, v)
                edge_labels[edge] = f'{data["flow"]}/{data["capacity"]}'
            if edge_labels:
                nx.draw_networkx_edge_labels(
                    graph,
                    pos,
                    edge_labels=edge_labels,
                    font_size=9,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8),
                )

        # Draw residual edges (only in residual mode)
        if mode == "residual":
            if residual_forward:
                nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=[(u, v) for u, v, _ in residual_forward],
                    edge_color="gray",
                    style="dotted",
                    arrows=True,
                    width=1.8,
                    alpha=0.6,
                    arrowsize=15,
                    connectionstyle='arc3,rad=0.05',
                )
            if residual_reverse:
                nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=[(u, v) for u, v, _ in residual_reverse],
                    edge_color="orange",
                    style="dashed",
                    arrows=True,
                    width=2.0,
                    alpha=0.6,
                    arrowsize=15,
                    connectionstyle='arc3,rad=-0.05',
                )
            if residual_forward or residual_reverse:
                residual_labels = {}
                for data_list in (residual_forward, residual_reverse):
                    for u, v, data in data_list:
                        residual_labels[(u, v)] = f'{data["capacity"]}'
                if residual_labels:
                    nx.draw_networkx_edge_labels(
                        graph, pos, edge_labels=residual_labels, font_color="#475569", font_size=8
                    )

        # Level labels
        level_map = self._get_level_map()
        for level_idx, nodes in level_map.items():
            for node in nodes:
                if node in pos:
                    x, y = pos[node]
                    if level_idx >= 0:
                        plt.text(x, y + 0.12, f"L{level_idx}", fontsize=8, color='gray', ha='center')
                    else:
                        plt.text(x, y + 0.12, "∞", fontsize=8, color='lightgray', ha='center')

        # Legend
        legend_handles = [
            Line2D([0], [0], color="gray", lw=2, label="Unused Edge"),
            Line2D([0], [0], color="royalblue", lw=2, label="Flowing Edge"),
            Line2D([0], [0], color="firebrick", lw=2, label="Saturated Edge"),
            Line2D([0], [0], color="gray", lw=2, linestyle=":", label="Residual Capacity"),
            Line2D([0], [0], color="orange", lw=2, linestyle="--", label="Reverse Residual"),
        ]
        if highlight_path:
            legend_handles.append(Line2D([0], [0], color="deepskyblue", lw=3.5, label="Augmenting Path"))
        plt.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9)
        plt.title(title, fontsize=13, pad=20)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()

    def _flow_distribution(self) -> List[Tuple[int, int, int, int]]:
        """Return list of (u, v, flow, capacity) for original edges only."""
        dist: List[Tuple[int, int, int, int]] = []
        for u in range(self.num_vertices):
            for e in self.graph[u]:
                if e.is_original:
                    dist.append((u, e.to, e.flow, e.capacity))
        order = {(u, v, c): i for i, (u, v, c) in enumerate(self.original_edges)}
        dist.sort(key=lambda x: order.get((x[0], x[1], x[3]), 0))
        return dist

    def run(self) -> Tuple[int, float]:
        """Run Dinic's algorithm with per-path iteration visualization."""
        self._reset_visual_outputs()
        self.iteration_metrics = []
        self.iteration_index = 0
        self.total_augmenting_paths = 0
        self.bfs_phases = 0
        self.bfs_time_total = 0.0
        self.dfs_time_total = 0.0
        total_flow = 0

        # Save initial graph
        initial_path = os.path.join(self.graph_visuals_dir, "initial_graph.png")
        self._save_graph_visual(initial_path, title="Initial Flow Network")

        start = time.perf_counter()
        self.logger.info("Starting Dinic's algorithm: source=%s sink=%s", self.source, self.sink)

        while True:
            # BFS to build level graph
            bfs_start = time.perf_counter()
            if not self._bfs_level_graph():
                break
            bfs_time = time.perf_counter() - bfs_start
            self.bfs_phases += 1
            self.bfs_time_total += bfs_time

            # Reset iterator pointers for DFS
            self.it_ptr = [0] * self.num_vertices

            # Find all augmenting paths in this level graph
            paths_in_phase = 0
            while True:
                # Find one path
                dfs_path_start = time.perf_counter()
                path_flow, path_nodes = self._dfs_find_path(self.source, float("inf"), [self.source])
                dfs_path_time = time.perf_counter() - dfs_path_start
                self.dfs_time_total += dfs_path_time
                
                if path_flow <= 0:
                    break

                self.iteration_index += 1
                self.total_augmenting_paths += 1
                paths_in_phase += 1
                iteration_dir = os.path.join(self.graph_visuals_dir, f"iteration_{self.iteration_index}")
                os.makedirs(iteration_dir, exist_ok=True)

                # Save residual_before.png
                residual_before_path = os.path.join(iteration_dir, "initial_residual.png")
                self._save_graph_visual(
                    residual_before_path,
                    title=f"Iteration {self.iteration_index}: Residual Network (Before)",
                    mode="residual",
                )

                # Save augmented_path.png (highlight the path)
                augmented_path_file = os.path.join(iteration_dir, "selected_augmented_path.png")
                self._save_graph_visual(
                    augmented_path_file,
                    title=f"Iteration {self.iteration_index}: Augmenting Path",
                    highlight_path=path_nodes,
                )

                # Apply the path
                self._apply_path(path_nodes, path_flow)
                total_flow += path_flow

                # Save residual_after.png
                residual_after_path = os.path.join(iteration_dir, "final_residual.png")
                self._save_graph_visual(
                    residual_after_path,
                    title=f"Iteration {self.iteration_index}: Residual Network (After)",
                    mode="residual",
                )

                # Log and record metrics
                path_str = " -> ".join(map(str, path_nodes))
                self.logger.info(
                    "Iteration %d: Path %s added flow %d (total: %d)",
                    self.iteration_index,
                    path_str,
                    path_flow,
                    total_flow,
                )

                self.iteration_metrics.append(
                    {
                        "iteration": self.iteration_index,
                        "path": path_nodes,
                        "flow_added": path_flow,
                        "total_flow": total_flow,
                        "bfs_time": bfs_time if paths_in_phase == 1 else 0,
                        "dfs_time": dfs_path_time,
                    }
                )

        runtime = time.perf_counter() - start

        # Save final graph
        final_path = os.path.join(self.graph_visuals_dir, "final_flow_graph.png")
        self._save_graph_visual(final_path, title="Final Max Flow Network")
        
        # Write iteration log CSV
        self._write_iteration_log()
        
        self.logger.info("Algorithm complete. Max flow: %s in %.6fs", total_flow, runtime)
        return total_flow, runtime

    def _write_iteration_log(self) -> None:
        """Write iteration_log.csv inside the graph's visuals folder."""
        log_path = os.path.join(self.graph_visuals_dir, "iteration_log.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "bfs_time", "dfs_time", "flow_added", "total_flow_after", "num_augmenting_paths"])
            for metric in self.iteration_metrics:
                writer.writerow([
                    metric["iteration"],
                    f"{metric.get('bfs_time', 0):.6f}",
                    f"{metric.get('dfs_time', 0):.6f}",
                    metric["flow_added"],
                    metric["total_flow"],
                    1,  # One path per iteration
                ])

    def write_outputs(self, max_flow: int, runtime: float) -> None:
        """Write output.txt with flow distribution and iteration details."""
        output_path = os.path.join(self.results_dir, "output.txt")
        lines = []
        lines.append(f"Maximum Flow: {max_flow}")
        lines.append("")
        lines.append("Flow Distribution:")
        for u, v, flow, cap in self._flow_distribution():
            arrow = "→"
            lines.append(f"{u} {arrow} {v} : {flow} / {cap}")
        lines.append("")
        lines.append(f"Runtime: {runtime:.4f}s")
        if self.iteration_metrics:
            lines.append("")
            lines.append("Iterations:")
            for metric in self.iteration_metrics:
                path_str = " -> ".join(map(str, metric["path"]))
                lines.append(
                    f"Iteration {metric['iteration']}: Path {path_str} (+{metric['flow_added']}, total {metric['total_flow']})"
                )
        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def append_metrics(self, graph_name: str, max_flow: int, runtime: float, family: str = "", seed: int = 0, trial: int = 0) -> None:
        """Append summary metrics to performance.csv."""
        summary_path = os.path.join(self.results_dir, "performance.csv")
        header = [
            "graph_filename", "family", "n", "m", "seed", "trial", "max_flow", "total_time",
            "bfs_time_total", "dfs_time_total", "num_iterations", "num_augmenting_paths"
        ]
        if os.path.exists(summary_path):
            with open(summary_path, newline="") as existing:
                reader = list(csv.reader(existing))
            existing_header = reader[0] if reader else []
            if existing_header != header:
                rows = reader[1:] if reader else []
                with open(summary_path, "w", newline="") as rewritten:
                    writer = csv.writer(rewritten)
                    writer.writerow(header)
                    for row in rows:
                        padded = row + [""] * (len(header) - len(existing_header))
                        writer.writerow(padded[: len(header)])
        file_exists = os.path.exists(summary_path)
        with open(summary_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(
                [
                    graph_name,
                    family,
                    self.num_vertices,
                    self.num_edges_declared,
                    seed,
                    trial,
                    max_flow,
                    f"{runtime:.6f}",
                    f"{self.bfs_time_total:.6f}",
                    f"{self.dfs_time_total:.6f}",
                    self.bfs_phases,
                    self.total_augmenting_paths,
                ]
            )

    def write_iteration_metrics(self, graph_name: str) -> None:
        """Write detailed per-iteration metrics to performance_iterations.csv."""
        detailed_path = os.path.join(self.results_dir, "performance_iterations.csv")
        file_exists = os.path.exists(detailed_path)
        with open(detailed_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "graph",
                        "iteration",
                        "flow_added",
                        "total_flow",
                        "path",
                    ]
                )
            for metric in self.iteration_metrics:
                path_str = " -> ".join(map(str, metric["path"]))
                writer.writerow(
                    [
                        graph_name,
                        metric["iteration"],
                        metric["flow_added"],
                        metric["total_flow"],
                        path_str,
                    ]
                )
