from collections import deque
import numpy as np

FREE, S_TREE, T_TREE = 0, 1, -1

class BoykovKolmogorov:
    def __init__(self, n):
        self.n = n
        self.cap = [{} for _ in range(n)]
        self.flow = [{} for _ in range(n)]
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v, c):
        if v not in self.cap[u]:
            self.cap[u][v] = 0
            self.cap[v][u] = 0
            self.flow[u][v] = 0
            self.flow[v][u] = 0
            self.adj[u].append(v)
            self.adj[v].append(u)
        self.cap[u][v] += c

    def residual(self, u, v):
        return self.cap[u][v] - self.flow[u][v]

    def max_flow(self, s, t):
        tree = [FREE] * self.n
        parent = [-1] * self.n
        active = deque()
        orphan = deque()

        tree[s] = S_TREE
        tree[t] = T_TREE
        active.append(s)
        active.append(t)

        def is_root(v):
            return v == s or v == t

        total_flow = 0

        while True:
            meet = -1

            # -------- GROWTH --------
            while active and meet == -1:
                p = active.popleft()
                for q in self.adj[p]:
                    if self.residual(p, q) > 0:
                        if tree[q] == FREE:
                            tree[q] = tree[p]
                            parent[q] = p
                            active.append(q)
                        elif tree[q] == -tree[p]:
                            meet = (p, q)
                            break

            if meet == -1:
                break

            # -------- AUGMENT --------
            p, q = meet
            bottleneck = float("inf")

            # path to source
            v = p
            while not is_root(v):
                u = parent[v]
                bottleneck = min(bottleneck, self.residual(u, v))
                v = u
            bottleneck = min(bottleneck, self.residual(s, v)) if v != s else bottleneck

            # path to sink
            v = q
            while not is_root(v):
                u = parent[v]
                bottleneck = min(bottleneck, self.residual(v, u))
                v = u
            bottleneck = min(bottleneck, self.residual(v, t)) if v != t else bottleneck

            # push flow
            v = p
            while not is_root(v):
                u = parent[v]
                self.flow[u][v] += bottleneck
                self.flow[v][u] -= bottleneck
                if self.residual(u, v) == 0:
                    orphan.append(v)
                v = u

            v = q
            while not is_root(v):
                u = parent[v]
                self.flow[v][u] += bottleneck
                self.flow[u][v] -= bottleneck
                if self.residual(v, u) == 0:
                    orphan.append(v)
                v = u

            total_flow += bottleneck

            # -------- ADOPTION --------
            while orphan:
                v = orphan.popleft()
                parent[v] = -1
                found = False

                for u in self.adj[v]:
                    if tree[u] == tree[v] and self.residual(u, v) > 0:
                        parent[v] = u
                        found = True
                        break

                if not found:
                    tree[v] = FREE
                    for u in self.adj[v]:
                        if parent[u] == v:
                            orphan.append(u)

        return total_flow

def build_grid_graph(H, W):
    N = H * W + 2
    S, T = 0, 1

    def nid(i, j):
        return 2 + i * W + j

    g = BoykovKolmogorov(N)

    # Unary (data) terms
    for i in range(H):
        for j in range(W):
            u = nid(i, j)
            g.add_edge(S, u, np.random.rand() * 5)   # foreground cost
            g.add_edge(u, T, np.random.rand() * 5)   # background cost

    # Pairwise (smoothness) terms: 4-neighbour grid
    for i in range(H):
        for j in range(W):
            u = nid(i, j)
            if i + 1 < H:
                v = nid(i + 1, j)
                w = np.random.rand() * 3
                g.add_edge(u, v, w)
                g.add_edge(v, u, w)
            if j + 1 < W:
                v = nid(i, j + 1)
                w = np.random.rand() * 3
                g.add_edge(u, v, w)
                g.add_edge(v, u, w)

    return g, S, T