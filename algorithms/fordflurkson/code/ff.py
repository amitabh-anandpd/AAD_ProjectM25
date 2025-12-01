"""
Ford–Fulkerson implementation with DFS (default) and BFS (Edmonds–Karp) option.

API mirrors Dinic's `Dinics` class where possible: constructor signature,
`run(callback)` returns `(max_flow, path_history)` where `path_history` is a
list of `(iteration, path_nodes, flow_added)` tuples.

This implementation stores the residual graph as adjacency maps `residual[u][v]`
and tracks `original_edges` and `flow` for original edges. Pushing flow updates
the residuals using the implicit reverse-edge rule (residual[v][u] += f).
"""
from collections import deque
import time
from typing import List, Tuple, Dict, Any


class FordFulkerson:
    def __init__(self, num_vertices: int, edges: List[Tuple[int, int, int]], source: int, sink: int, method: str = "dfs"):
        self.num_vertices = num_vertices
        self.source = source
        self.sink = sink
        self.method = method.lower()

        # Residual capacities: dict of dicts
        self.residual: Dict[int, Dict[int, int]] = {i: {} for i in range(num_vertices)}
        # Keep original capacities for final flow distribution and min-cut
        self.original_capacity: Dict[Tuple[int, int], int] = {}
        self.original_edges: List[Tuple[int, int, int]] = []
        # Flow on original edges
        self.flow: Dict[Tuple[int, int], int] = {}

        for u, v, c in edges:
            # record original
            self.original_edges.append((u, v, c))
            self.original_capacity[(u, v)] = c
            self.flow[(u, v)] = 0
            # initialize residual forward and reverse
            self.residual[u].setdefault(v, 0)
            self.residual[v].setdefault(u, 0)
            self.residual[u][v] += c

        # Metrics
        self.bfs_time_total = 0.0
        self.dfs_time_total = 0.0
        self.num_dfs_calls = 0
        self.total_augmenting_paths = 0

    def _serialize_residual(self) -> List[Tuple[int, int, int]]:
        out = []
        for u in range(self.num_vertices):
            for v, cap in self.residual[u].items():
                if cap > 0:
                    out.append((u, v, cap))
        return out

    def _dfs_find_path(self) -> Tuple[List[int], int, List[int]]:
        # Returns (path_nodes, bottleneck, stack_history)
        start = time.perf_counter()
        visited = [False] * self.num_vertices
        stack_history: List[int] = []

        def dfs(u: int, t: int, bottleneck: int, path: List[int]) -> Tuple[int, List[int]]:
            self.num_dfs_calls += 1
            visited[u] = True
            stack_history.append(u)
            if u == t:
                return bottleneck, path.copy()

            for v, cap in list(self.residual[u].items()):
                if cap <= 0 or visited[v]:
                    continue
                path.append(v)
                pushed, found_path = dfs(v, t, min(bottleneck, cap), path)
                if pushed > 0:
                    path.pop()
                    return pushed, found_path
                path.pop()

            return 0, []

        started_path = [self.source]
        pushed, found_path = dfs(self.source, self.sink, float("inf"), started_path)
        self.dfs_time_total += time.perf_counter() - start
        if pushed == float("inf"):
            pushed = 0
        return found_path, int(pushed), stack_history

    def _bfs_find_path(self) -> Tuple[List[int], int, List[int]]:
        # Edmonds–Karp shortest augmenting path via BFS
        start = time.perf_counter()
        parent = [-1] * self.num_vertices
        parent_edge = [-1] * self.num_vertices
        q = deque([self.source])
        visited = [False] * self.num_vertices
        visited[self.source] = True
        order = []

        while q:
            u = q.popleft()
            order.append(u)
            for v, cap in self.residual[u].items():
                if cap > 0 and not visited[v]:
                    visited[v] = True
                    parent[v] = u
                    q.append(v)
                    if v == self.sink:
                        q.clear()
                        break

        self.bfs_time_total += time.perf_counter() - start

        if not visited[self.sink]:
            return [], 0, order

        # Reconstruct path
        path = []
        v = self.sink
        bottleneck = float("inf")
        while v != -1 and v != self.source:
            u = parent[v]
            if u == -1:
                break
            path.append(v)
            cap = self.residual[u].get(v, 0)
            bottleneck = min(bottleneck, cap)
            v = u
        path.append(self.source)
        path.reverse()
        return path, int(bottleneck if bottleneck != float("inf") else 0), order

    def _apply_path(self, path: List[int], f: int) -> None:
        # Update residuals
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # decrease forward residual
            self.residual[u][v] = self.residual[u].get(v, 0) - f
            # increase reverse residual
            self.residual[v][u] = self.residual[v].get(u, 0) + f

            # Update flow on original edges if present
            if (u, v) in self.flow:
                self.flow[(u, v)] += f
            elif (v, u) in self.flow:
                # we pushed on reverse of original edge -> subtract
                self.flow[(v, u)] -= f

    def run(self, callback=None) -> Tuple[int, List[Tuple[int, List[int], int]]]:
        total_flow = 0
        iteration = 0
        path_history: List[Tuple[int, List[int], int]] = []
        detailed_logs: List[Dict[str, Any]] = []

        while True:
            snapshot_before = self._serialize_residual()

            if self.method == "bfs":
                path, bottleneck, stack_history = self._bfs_find_path()
            else:
                path, bottleneck, stack_history = self._dfs_find_path()

            if not path or bottleneck <= 0:
                break

            iteration += 1
            # Apply path
            self._apply_path(path, bottleneck)
            total_flow += bottleneck
            self.total_augmenting_paths += 1

            snapshot_after = self._serialize_residual()

            # Record minimal path tuple for compatibility
            path_history.append((iteration, path, bottleneck))

            # Detailed per-iteration log
            detailed_logs.append({
                "iteration": iteration,
                "path": path,
                "bottleneck": bottleneck,
                "flow_added": bottleneck,
                "residual_snapshot_before": snapshot_before,
                "residual_snapshot_after": snapshot_after,
                "stack_history": stack_history,
            })

            # Callback (compatible signature with Dinic runner)
            if callback:
                callback(iteration, iteration, path, bottleneck, total_flow, False)

        return total_flow, path_history

    def get_flow_distribution(self) -> List[Tuple[int, int, int, int]]:
        out = []
        for u, v, c in self.original_edges:
            f = int(self.flow.get((u, v), 0))
            out.append((u, v, f, c))
        return out

    def get_min_cut(self) -> Tuple[int, set, set, List[Tuple[int, int, int]]]:
        # BFS on residual (capacities > 0)
        reachable = set()
        q = deque([self.source])
        reachable.add(self.source)
        while q:
            u = q.popleft()
            for v, cap in self.residual[u].items():
                if cap > 0 and v not in reachable:
                    reachable.add(v)
                    q.append(v)

        S = reachable
        T = set(range(self.num_vertices)) - reachable

        cut_edges = []
        cut_value = 0
        for (u, v, c) in self.original_edges:
            if u in S and v in T and c > 0:
                cut_edges.append((u, v, c))
                cut_value += c

        return cut_value, S, T, cut_edges


def example_usage():
    # Simple example for manual testing
    edges = [(0, 1, 3), (1, 2, 2), (0, 2, 1)]
    ff = FordFulkerson(3, edges, 0, 2)
    maxflow, history = ff.run()
    print("maxflow=", maxflow)


if __name__ == "__main__":
    example_usage()
