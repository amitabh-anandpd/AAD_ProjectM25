"""
Pure Dinic's algorithm implementation for maximum flow.

This module contains only the core algorithm logic:
- BFS level graph construction
- DFS blocking flow discovery
- Residual graph updates
- Reverse edge handling
- Minimum cut extraction

No visualization, metrics collection, or file I/O happens here.
"""
from collections import deque
from typing import List, Tuple


class Edge:
    """
    Represents an edge in the flow network.
    
    In this class, I store edge information including destination, reverse edge index,
    capacity, and current flow. The residual capacity is computed as capacity - flow.
    """
    def __init__(self, to: int, rev: int, capacity: int, is_original: bool):
        """
        Initialize an edge.
        
        Args:
            to: Destination vertex
            rev: Index of reverse edge in the destination's adjacency list
            capacity: Edge capacity
            is_original: True if this is an original edge (not a reverse residual edge)
        """
        self.to = to
        self.rev = rev
        self.capacity = capacity
        self.flow = 0
        self.is_original = is_original

    @property
    def residual_capacity(self) -> int:
        """Compute residual capacity (remaining capacity for flow)."""
        return self.capacity - self.flow


class Dinics:
    """
    Pure implementation of Dinic's algorithm for maximum flow.
    
    This class implements the core algorithm:
    1. Build level graph using BFS
    2. Find blocking flow using DFS
    3. Update residual capacities
    4. Repeat until no augmenting path exists
    
    Here I track timing metrics (BFS, DFS) and iteration counts for performance analysis.
    """
    
    def __init__(self, num_vertices: int, edges: List[Tuple[int, int, int]], 
                 source: int, sink: int):
        """
        Initialize Dinic's algorithm with a graph.
        
        Args:
            num_vertices: Number of vertices in the graph
            edges: List of (u, v, capacity) tuples
            source: Source vertex index
            sink: Sink vertex index
        """
        self.num_vertices = num_vertices
        self.source = source
        self.sink = sink
        
        # Graph representation: adjacency list of Edge objects
        self.graph: List[List[Edge]] = [[] for _ in range(num_vertices)]
        self.original_edges: List[Tuple[int, int, int]] = []
        
        # Algorithm state
        self.level: List[int] = []  # Level assignment for each vertex
        self.it_ptr: List[int] = []  # Iterator pointers for DFS
        
        # Build graph from edges
        for u, v, capacity in edges:
            self.add_edge(u, v, capacity)
            self.original_edges.append((u, v, capacity))
        
        # Metrics tracking
        self.total_augmenting_paths = 0
        self.bfs_phases = 0
        self.bfs_time_total = 0.0
        self.dfs_time_total = 0.0

    def add_edge(self, u: int, v: int, capacity: int) -> None:
        """
        Add an edge with its reverse residual edge.
        
        In this function, I create both the forward edge and its reverse edge.
        The reverse edge is needed for residual graph operations and starts with
        capacity 0. Each edge stores a reference to its reverse edge.
        """
        # Forward edge (original direction)
        forward = Edge(to=v, rev=len(self.graph[v]), capacity=capacity, is_original=True)
        # Backward edge (reverse direction for residual graph)
        backward = Edge(to=u, rev=len(self.graph[u]), capacity=0, is_original=False)
        
        self.graph[u].append(forward)
        # Update reverse edge index after appending
        backward.rev = len(self.graph[u]) - 1
        self.graph[v].append(backward)

    def _bfs_level_graph(self) -> bool:
        """
        Build level graph using BFS.
        
        Here I perform a BFS from the source to assign levels to all reachable vertices.
        The level of a vertex is its distance from the source in the residual graph,
        considering only edges with positive residual capacity.
        
        Returns:
            True if sink is reachable, False otherwise
        """
        import time
        bfs_start = time.perf_counter()
        
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
        
        self.bfs_time_total += time.perf_counter() - bfs_start
        return self.level[self.sink] >= 0

    def _dfs_find_path(self, u: int, pushed: int, path: List[int]) -> Tuple[int, List[int]]:
        """
        Find a single augmenting path using DFS.
        
        In this function, I recursively explore the level graph to find an augmenting
        path from current vertex to sink. I only traverse edges that:
        1. Have positive residual capacity
        2. Lead to a vertex at the next level (level[to] == level[u] + 1)
        
        Args:
            u: Current vertex
            pushed: Maximum flow that can be pushed so far
            path: Current path being explored
            
        Returns:
            Tuple of (flow_value, path_nodes) where path_nodes is the augmenting path
        """
        import time
        dfs_start = time.perf_counter()
        
        if u == self.sink:
            self.dfs_time_total += time.perf_counter() - dfs_start
            return pushed, path.copy()
        
        while self.it_ptr[u] < len(self.graph[u]):
            e = self.graph[u][self.it_ptr[u]]
            # Only follow edges with residual capacity and correct level
            if e.residual_capacity > 0 and self.level[e.to] == self.level[u] + 1:
                path.append(e.to)
                # Recursively explore from next vertex
                flow, found_path = self._dfs_find_path(e.to, min(pushed, e.residual_capacity), path)
                if flow > 0:
                    path.pop()
                    self.dfs_time_total += time.perf_counter() - dfs_start
                    return flow, found_path
                path.pop()
            self.it_ptr[u] += 1
        
        self.dfs_time_total += time.perf_counter() - dfs_start
        return 0, []

    def _apply_path(self, path: List[int], flow: int) -> None:
        """
        Apply flow along an augmenting path.
        
        This block updates the flow values along the path and also updates
        the reverse edges to maintain the residual graph invariant.
        """
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # Find and update the forward edge
            for e in self.graph[u]:
                if e.to == v:
                    e.flow += flow
                    # Update reverse edge (subtract flow)
                    self.graph[v][e.rev].flow -= flow
                    break

    def run(self, callback=None) -> Tuple[int, List[Tuple[int, List[int], int]]]:
        """
        Run Dinic's algorithm to find maximum flow.
        
        The algorithm repeatedly:
        1. Builds a level graph using BFS
        2. Finds all augmenting paths in the level graph using DFS
        3. Applies flow along each path
        4. Repeats until no augmenting path exists
        
        Args:
            callback: Optional callback function called after each path is found.
                     Signature: callback(iteration, path, flow_added, total_flow, is_new_bfs_phase)
        
        Returns:
            Tuple of (max_flow, path_history) where path_history is a list of
            (iteration, path, flow_added) tuples for each augmenting path found
        """
        total_flow = 0
        path_history: List[Tuple[int, List[int], int]] = []
        iteration = 0
        path_index = 0

        while True:
            # Build level graph
            if not self._bfs_level_graph():
                # No path from source to sink
                break
            
            self.bfs_phases += 1
            iteration += 1
            is_new_bfs_phase = True

            # Reset iterator pointers for DFS
            self.it_ptr = [0] * self.num_vertices

            # Find all augmenting paths in this level graph
            while True:
                path_flow, path_nodes = self._dfs_find_path(self.source, float("inf"), [self.source])
                
                if path_flow <= 0:
                    break

                path_index += 1
                
                # Apply the augmenting path
                self._apply_path(path_nodes, path_flow)
                total_flow += path_flow
                self.total_augmenting_paths += 1
                
                # Record path for history
                path_history.append((iteration, path_nodes, path_flow))
                
                # Call callback if provided
                if callback:
                    callback(iteration, path_index, path_nodes, path_flow, total_flow, is_new_bfs_phase)
                
                is_new_bfs_phase = False
        
        return total_flow, path_history
    
    def get_flow_distribution(self) -> List[Tuple[int, int, int, int]]:
        """
        Get current flow distribution for all original edges.
        
        Returns:
            List of (u, v, flow, capacity) tuples for each original edge
        """
        flow_dist: List[Tuple[int, int, int, int]] = []
        for u in range(self.num_vertices):
            for e in self.graph[u]:
                if e.is_original:
                    flow_dist.append((u, e.to, e.flow, e.capacity))
        
        # Sort by original edge order
        order = {(u, v, c): i for i, (u, v, c) in enumerate(self.original_edges)}
        flow_dist.sort(key=lambda x: order.get((x[0], x[1], x[3]), 0))
        return flow_dist
    
    def get_residual_edges(self) -> List[Tuple[int, int, int, bool]]:
        """
        Get all edges with positive residual capacity.
        
        Returns:
            List of (u, v, residual_capacity, is_original) tuples
        """
        residual = []
        for u in range(self.num_vertices):
            for e in self.graph[u]:
                cap = e.residual_capacity
                if cap > 0:
                    residual.append((u, e.to, cap, e.is_original))
        return residual
    
    def get_min_cut(self) -> Tuple[int, set, set, List[Tuple[int, int]]]:
        """
        Extract minimum cut after maximum flow has been computed.
        
        This block performs a BFS on the final residual graph to determine which
        vertices are reachable from the source. The minimum cut consists of edges
        from reachable vertices (S) to unreachable vertices (T) that are saturated.
        
        Returns:
            Tuple of (cut_value, S (reachable vertices), T (unreachable vertices), 
            cut_edges) where cut_edges is a list of (u, v) tuples
        """
        # BFS on residual graph to find reachable vertices
        reachable = set()
        queue: deque[int] = deque([self.source])
        reachable.add(self.source)
        
        while queue:
            u = queue.popleft()
            for e in self.graph[u]:
                if e.residual_capacity > 0 and e.to not in reachable:
                    reachable.add(e.to)
                    queue.append(e.to)
        
        # S = reachable vertices, T = unreachable vertices
        S = reachable
        T = set(range(self.num_vertices)) - reachable
        
        # Find edges from S to T that are in the original graph and saturated
        cut_edges: List[Tuple[int, int]] = []
        cut_value = 0
        
        for u in S:
            for e in self.graph[u]:
                if e.to in T and e.is_original:
                    # This edge crosses the cut
                    cut_edges.append((u, e.to))
                    # Cut value is the capacity of saturated edges from S to T
                    cut_value += e.capacity
        
        return cut_value, S, T, cut_edges
