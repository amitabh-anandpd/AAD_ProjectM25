import time
from collections import deque

class Edge:
    def __init__(self, to, rev, capacity):
        self.to = to
        self.rev = rev
        self.capacity = capacity
        self.flow = 0

class PushRelabel:
    def __init__(self, graph_dict, source, sink):
        self.source = source
        self.sink = sink
        self.num_vertices = len(graph_dict)
        self.graph = [[] for _ in range(self.num_vertices)]
        self.excess = [0] * self.num_vertices
        self.height = [0] * self.num_vertices
        self.active = deque()
        
        # GAP HEURISTIC: Track count of nodes at each height
        # Max height can be 2*N theoretically
        self.height_counts = [0] * (2 * self.num_vertices + 7) 

        for u, neighbors in graph_dict.items():
            for v, cap in neighbors.items():
                a = Edge(v, len(self.graph[v]), cap)
                b = Edge(u, len(self.graph[u]), 0)
                self.graph[u].append(a)
                self.graph[v].append(b)

    def _push(self, u, edge):
        val = min(self.excess[u], edge.capacity - edge.flow)
        edge.flow += val
        self.graph[edge.to][edge.rev].flow -= val
        self.excess[u] -= val
        self.excess[edge.to] += val
        
        if val > 0 and self.excess[edge.to] == val and edge.to != self.source and edge.to != self.sink:
            self.active.append(edge.to)

    def _relabel(self, u):
        """
        Relabels u to the lowest height that allows a push + GAP HEURISTIC.
        """
        old_height = self.height[u]
        
        # --- GAP HEURISTIC START ---
        # If u was the ONLY node at old_height, a "gap" has formed.
        # Nodes above this gap are now disconnected from the sink.
        # We can lift them all to N + 1 instantly.
        if self.height_counts[old_height] == 1:
            for i in range(self.num_vertices):
                if self.height[i] >= old_height and self.height[i] < self.num_vertices:
                    self.height_counts[self.height[i]] -= 1
                    self.height[i] = self.num_vertices + 1 # Lift to "dead" zone
                    self.height_counts[self.height[i]] += 1
            # We return early because u has been lifted by the loop above
            return
        # --- GAP HEURISTIC END ---

        # Standard Relabel Logic
        min_height = float('inf')
        for edge in self.graph[u]:
            if edge.capacity - edge.flow > 0:
                min_height = min(min_height, self.height[edge.to])
        
        if min_height < float('inf'):
            # Update counts
            self.height_counts[old_height] -= 1
            self.height[u] = min_height + 1
            self.height_counts[self.height[u]] += 1

    def _discharge(self, u):
        while self.excess[u] > 0:
            # Check if node was lifted by gap heuristic logic from another node
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
                # If relabel created a gap and lifted u, loop breaks next iteration

    def run(self):
        start_time = time.perf_counter()
        
        # Initialize Heights
        self.height[self.source] = self.num_vertices
        
        # Initialize Height Counts
        # Source is at height N (1 node)
        # All others are at height 0 (N-1 nodes)
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
                
        # Calculate Max Flow
        max_flow = 0
        for edge in self.graph[self.source]:
            max_flow += edge.flow

        end_time = time.perf_counter()
        return max_flow, end_time - start_time
    
    def get_min_cut_source_side(self):
        """BFS on residual graph to find S-set."""
        visited = set()
        queue = deque([self.source])
        visited.add(self.source)
        while queue:
            u = queue.popleft()
            for edge in self.graph[u]:
                if edge.capacity - edge.flow > 0 and edge.to not in visited:
                    visited.add(edge.to)
                    queue.append(edge.to)
        return visited