import time
from collections import deque

class Edge:
    def __init__(self, to, rev, capacity):
        self.to = to
        self.rev = rev
        self.capacity = capacity
        self.flow = 0

class PushRelabel:
    """
    Push-Relabel algorithm with Gap Heuristic and Min-Cut extraction.
    """
    def __init__(self, graph_dict, source, sink):
        self.source = source
        self.sink = sink
        self.num_vertices = len(graph_dict)
        self.graph = [[] for _ in range(self.num_vertices)]
        self.excess = [0] * self.num_vertices
        self.height = [0] * self.num_vertices
        self.active = deque()
        
        # GAP HEURISTIC data
        self.height_counts = [0] * (2 * self.num_vertices + 7) 

        # Build Internal Graph Structure
        # Note: We store original capacity to calculate Min-Cut later
        self.original_edges = [] 

        for u, neighbors in graph_dict.items():
            for v, cap in neighbors.items():
                # Add forward edge
                a = Edge(v, len(self.graph[v]), cap)
                # Add backward edge
                b = Edge(u, len(self.graph[u]), 0)
                
                self.graph[u].append(a)
                self.graph[v].append(b)
                
                # Store reference for Min-Cut calculation: (u, v, capacity, edge_object)
                self.original_edges.append((u, v, cap, a))

    def _push(self, u, edge):
        val = min(self.excess[u], edge.capacity - edge.flow)
        edge.flow += val
        self.graph[edge.to][edge.rev].flow -= val
        self.excess[u] -= val
        self.excess[edge.to] += val
        
        if val > 0 and self.excess[edge.to] == val and edge.to != self.source and edge.to != self.sink:
            self.active.append(edge.to)

    def _relabel(self, u):
        old_height = self.height[u]
        
        # --- GAP HEURISTIC ---
        if self.height_counts[old_height] == 1:
            for i in range(self.num_vertices):
                if self.height[i] >= old_height and self.height[i] < self.num_vertices:
                    self.height_counts[self.height[i]] -= 1
                    self.height[i] = self.num_vertices + 1
                    self.height_counts[self.height[i]] += 1
            return
        
        min_height = float('inf')
        for edge in self.graph[u]:
            if edge.capacity - edge.flow > 0:
                min_height = min(min_height, self.height[edge.to])
        
        if min_height < float('inf'):
            self.height_counts[old_height] -= 1
            self.height[u] = min_height + 1
            self.height_counts[self.height[u]] += 1

    def _discharge(self, u):
        while self.excess[u] > 0:
            if self.height[u] >= self.num_vertices:
                break

            pushed_flag = False
            for edge in self.graph[u]:
                if edge.capacity - edge.flow > 0 and self.height[u] == self.height[edge.to] + 1:
                    self._push(u, edge)
                    pushed_flag = True
                    if self.excess[u] == 0:
                        break
            
            if not pushed_flag:
                self._relabel(u)

    def run(self):
        """
        Executes the Push-Relabel algorithm.
        Returns: (max_flow_value, runtime_seconds)
        """
        start_time = time.perf_counter()
        
        self.height[self.source] = self.num_vertices
        self.height_counts[0] = self.num_vertices - 1
        self.height_counts[self.num_vertices] = 1
        
        self.excess[self.source] = float('inf')
        
        # Preflow
        for edge in self.graph[self.source]:
            self._push(self.source, edge)
            
        while self.active:
            u = self.active.popleft()
            if u != self.source and u != self.sink:
                self._discharge(u)
                
        max_flow = 0
        for edge in self.graph[self.source]:
            max_flow += edge.flow

        end_time = time.perf_counter()
        return max_flow, end_time - start_time
    
    def get_min_cut(self):
        """
        Calculates the Minimum Cut based on the residual graph.
        Returns: (cut_value, reachable_nodes_set)
        """
        # 1. Find reachable nodes from Source in the residual graph (BFS)
        reachable = set()
        queue = deque([self.source])
        reachable.add(self.source)
        
        while queue:
            u = queue.popleft()
            for edge in self.graph[u]:
                # If there is residual capacity, we can traverse
                if edge.capacity - edge.flow > 0 and edge.to not in reachable:
                    reachable.add(edge.to)
                    queue.append(edge.to)
        
        # 2. Calculate Cut Value: Sum of capacities of edges going from Reachable -> Unreachable
        cut_value = 0
        cut_edges = []
        
        for u, v, cap, edge_obj in self.original_edges:
            if u in reachable and v not in reachable:
                cut_value += cap
                cut_edges.append((u, v, cap))
                
        return cut_value, reachable