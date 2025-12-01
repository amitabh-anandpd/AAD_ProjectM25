"""
Advanced graph generator for Dinic experiments.

This generator produces 4 families x 2 modes x N graphs each:
  families: general, unitcap, simple, worst
  modes: V (constant edges, increasing vertices), E (constant vertices, increasing edges)

Output folders (under project root):
  general_v/, general_e/, unitcap_v/, unitcap_e/, simple_v/, simple_e/, worst_v/, worst_e/

Each graph is saved as: graph_<family>_<mode>_<number>.txt

All graphs satisfy:
 - directed edges allowed
 - no self-loops
 - no duplicate edges
 - guaranteed s->t path (validated)
 - source = 0, sink = n-1

Reproducibility: uses random.seed(1000 + i) per graph.
"""
import os
import random
import math
from typing import List, Tuple, Set
from collections import deque

# ---------------------------------------------------------------------------
# Global parameters (updated as requested)
# ---------------------------------------------------------------------------
NUM_GRAPHS_PER_MODE = 20
BASE_VERTICES_V = 80       # starting vertices for V-mode
BASE_EDGES_E = 300         # starting edges for E-mode
VERTEX_INCREMENT = 20
EDGE_INCREMENT = 100
CONSTANT_EDGES_V = 300     # edges used in V-mode (constant)
CONSTANT_VERTICES_E = 150  # vertices used in E-mode (constant)
BASE_SEED = 1000

# Capacity ranges are handled inside family constructors where appropriate.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_graph(num_vertices: int, edges: List[Tuple[int, int, int]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"{num_vertices} {len(edges)}\n")
        for u, v, c in edges:
            f.write(f"{u} {v} {c}\n")


def has_path(n: int, edges: List[Tuple[int, int, int]]) -> bool:
    """Simple BFS to check reachability from 0 to n-1 in directed graph."""
    adj = [[] for _ in range(n)]
    for u, v, _ in edges:
        if 0 <= u < n and 0 <= v < n:
            adj[u].append(v)
    src, sink = 0, n - 1
    q = deque([src])
    seen = [False] * n
    seen[src] = True
    while q:
        u = q.popleft()
        if u == sink:
            return True
        for w in adj[u]:
            if not seen[w]:
                seen[w] = True
                q.append(w)
    return False


def add_random_edges(n: int, target_edges: int, edges: List[Tuple[int, int, int]],
                     edge_set: Set[Tuple[int, int]], capacity_fn) -> List[Tuple[int, int, int]]:
    """Add random edges up to target_edges without duplicates or self-loops.

    capacity_fn: function() -> capacity
    """
    attempts = 0
    max_attempts = max(10000, n * n * 2)
    while len(edges) < target_edges and attempts < max_attempts:
        u = random.randrange(0, n)
        v = random.randrange(0, n)
        if u == v:
            attempts += 1
            continue
        if (u, v) in edge_set:
            attempts += 1
            continue
        c = capacity_fn()
        if c <= 0:
            attempts += 1
            continue
        edges.append((u, v, c))
        edge_set.add((u, v))
        attempts += 1
    # truncate if overshot
    return edges[:target_edges]


# ---------------------------------------------------------------------------
# Family constructors
# Each returns (n, edges)
# ---------------------------------------------------------------------------

def construct_general(n: int, m: int) -> Tuple[int, List[Tuple[int, int, int]]]:
    """General graphs: random edges, capacities in [1,100], ensure s->t path."""
    edges: List[Tuple[int, int, int]] = []
    edge_set: Set[Tuple[int, int]] = set()

    # Ensure path: build random path from 0 to n-1
    if n < 2:
        raise ValueError("n must be >= 2")
    intermediate = list(range(1, n - 1))
    random.shuffle(intermediate)
    # choose a random length path but include sink
    path = [0]
    # ensure at least one intermediate sometimes
    k = min(len(intermediate), max(0, int(len(intermediate) * 0.2)))
    path.extend(intermediate[:k])
    path.append(n - 1)
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        c = random.randint(1, 100)
        edges.append((u, v, c))
        edge_set.add((u, v))

    # Add remaining edges randomly (moderate extra connections)
    def cap():
        return random.randint(1, 100)

    edges = add_random_edges(n, m, edges, edge_set, cap)
    return n, edges


def construct_unitcap(n: int, m: int) -> Tuple[int, List[Tuple[int, int, int]]]:
    """Unit capacity: layered graph, capacities = 1, edges only between adjacent layers."""
    if n < 2:
        raise ValueError("n must be >= 2")
    edges: List[Tuple[int, int, int]] = []
    edge_set: Set[Tuple[int, int]] = set()

    # Determine number of layers k ~ sqrt(n)
    k = max(2, int(math.sqrt(n)))
    # distribute nodes into k layers
    layers = [[] for _ in range(k)]
    layers[0].append(0)
    layers[-1].append(n - 1)
    mids = list(range(1, n - 1))
    for idx, v in enumerate(mids):
        layers[1 + (idx % (k - 2))].append(v)

    # Ensure at least one path: connect one node from each consecutive layer
    path_nodes = []
    for L in layers:
        # pick representative
        path_nodes.append(L[0])
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i + 1]
        edges.append((u, v, 1))
        edge_set.add((u, v))

    # Add extra edges only between adjacent layers
    attempts = 0
    max_attempts = max(10000, n * n * 2)
    while len(edges) < m and attempts < max_attempts:
        layer_i = random.randrange(0, k - 1)
        u = random.choice(layers[layer_i])
        v = random.choice(layers[layer_i + 1])
        if u == v or (u, v) in edge_set:
            attempts += 1
            continue
        edges.append((u, v, 1))
        edge_set.add((u, v))
        attempts += 1

    edges = edges[:m]
    return n, edges


def construct_simple(n: int, m: int) -> Tuple[int, List[Tuple[int, int, int]]]:
    """Simple: source->middle edges cap=1, middle->sink cap=1, internal random [1,100]"""
    if n < 3:
        raise ValueError("n must be >= 3 for simple graphs")
    edges: List[Tuple[int, int, int]] = []
    edge_set: Set[Tuple[int, int]] = set()

    source = 0
    sink = n - 1
    middle = list(range(1, n - 1))

    deg = max(1, int(math.sqrt(n)))
    # connect source -> random middle nodes with cap=1
    src_targets = random.sample(middle, min(deg, len(middle)))
    for v in src_targets:
        edges.append((source, v, 1))
        edge_set.add((source, v))

    # connect random middle nodes -> sink with cap=1
    sink_sources = random.sample(middle, min(deg, len(middle)))
    for u in sink_sources:
        if (u, sink) not in edge_set:
            edges.append((u, sink, 1))
            edge_set.add((u, sink))

    # Add internal middle->middle edges with random capacities
    def cap():
        return random.randint(1, 100)

    edges = add_random_edges(n, m, edges, edge_set, cap)
    return n, edges


def construct_worst(n: int, m: int) -> Tuple[int, List[Tuple[int, int, int]]]:
    """Approximate worst-case construction for Dinic: chain + misleading dense part. capacities=1"""
    if n < 3:
        raise ValueError("n must be >= 3 for worst graphs")
    edges: List[Tuple[int, int, int]] = []
    edge_set: Set[Tuple[int, int]] = set()

    source = 0
    sink = n - 1

    # Create a long chain (the one usable augmenting path)
    # chain length roughly n//3
    chain_len = max(2, n // 3)
    chain_nodes = [0]
    # pick chain internal nodes
    available = list(range(1, n - 1))
    random.shuffle(available)
    chain_internal = available[: chain_len - 1]
    chain_nodes.extend(chain_internal)
    chain_nodes.append(sink)
    for i in range(len(chain_nodes) - 1):
        u, v = chain_nodes[i], chain_nodes[i + 1]
        edges.append((u, v, 1))
        edge_set.add((u, v))

    # Add many edges that appear useful but do not shortcut the chain
    non_chain = [v for v in range(n) if v not in chain_nodes]

    attempts = 0
    max_attempts = max(20000, n * n * 4)
    while len(edges) < m and attempts < max_attempts:
        if random.random() < 0.7 and non_chain:
            u = random.choice(non_chain)
            v = random.choice(non_chain + chain_nodes)
        else:
            u = random.choice(chain_nodes)
            v = random.choice(non_chain + chain_nodes)
        if u == v or (u, v) in edge_set:
            attempts += 1
            continue
        # Avoid direct shortcuts from early chain positions to very late chain nodes
        if u in chain_nodes and v in chain_nodes:
            idx_u = chain_nodes.index(u)
            idx_v = chain_nodes.index(v)
            if idx_u + 1 < idx_v:
                if random.random() < 0.8:
                    attempts += 1
                    continue
        edges.append((u, v, 1))
        edge_set.add((u, v))
        attempts += 1

    edges = edges[:m]
    return n, edges


# ---------------------------------------------------------------------------
# Mode generators: each computes n/m and calls family constructors
# ---------------------------------------------------------------------------

def _write_named_graph(folder: str, family: str, mode: str, i: int, n: int, edges: List[Tuple[int, int, int]]):
    project_root = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(project_root, folder)
    os.makedirs(out_dir, exist_ok=True)
    familyname = f"{family}_{mode}"
    # Name format requested: familyname_number (no extra prefix)
    filename = f"{familyname}_{i}.txt"
    save_graph(n, edges, os.path.join(out_dir, filename))


def generate_general_v():
    folder = "general_v"
    family = "general"
    mode = "v"
    for i in range(1, NUM_GRAPHS_PER_MODE + 1):
        random.seed(BASE_SEED + i)
        n = BASE_VERTICES_V + (i - 1) * VERTEX_INCREMENT
        m = CONSTANT_EDGES_V
        n_out, edges = construct_general(n, m)
        edges = edges[:m]
        familyname = f"{family}_{mode}"
        if not has_path(n_out, edges):
            raise ValueError(f"Generated {familyname}_{i} has no valid s->t path")
        _write_named_graph(folder, family, mode, i, n_out, edges)


def generate_general_e():
    folder = "general_e"
    family = "general"
    mode = "e"
    for i in range(1, NUM_GRAPHS_PER_MODE + 1):
        random.seed(BASE_SEED + i)
        n = CONSTANT_VERTICES_E
        m = BASE_EDGES_E + (i - 1) * EDGE_INCREMENT
        n_out, edges = construct_general(n, m)
        edges = edges[:m]
        familyname = f"{family}_{mode}"
        if not has_path(n_out, edges):
            raise ValueError(f"Generated {familyname}_{i} has no valid s->t path")
        _write_named_graph(folder, family, mode, i, n_out, edges)


def generate_unitcap_v():
    folder = "unitcap_v"
    family = "unitcap"
    mode = "v"
    for i in range(1, NUM_GRAPHS_PER_MODE + 1):
        random.seed(BASE_SEED + i)
        n = BASE_VERTICES_V + (i - 1) * VERTEX_INCREMENT
        m = CONSTANT_EDGES_V
        n_out, edges = construct_unitcap(n, m)
        edges = edges[:m]
        familyname = f"{family}_{mode}"
        if not has_path(n_out, edges):
            raise ValueError(f"Generated {familyname}_{i} has no valid s->t path")
        _write_named_graph(folder, family, mode, i, n_out, edges)


def generate_unitcap_e():
    folder = "unitcap_e"
    family = "unitcap"
    mode = "e"
    for i in range(1, NUM_GRAPHS_PER_MODE + 1):
        random.seed(BASE_SEED + i)
        n = CONSTANT_VERTICES_E
        m = BASE_EDGES_E + (i - 1) * EDGE_INCREMENT
        n_out, edges = construct_unitcap(n, m)
        edges = edges[:m]
        familyname = f"{family}_{mode}"
        if not has_path(n_out, edges):
            raise ValueError(f"Generated {familyname}_{i} has no valid s->t path")
        _write_named_graph(folder, family, mode, i, n_out, edges)


def generate_simple_v():
    folder = "simple_v"
    family = "simple"
    mode = "v"
    for i in range(1, NUM_GRAPHS_PER_MODE + 1):
        random.seed(BASE_SEED + i)
        n = BASE_VERTICES_V + (i - 1) * VERTEX_INCREMENT
        m = CONSTANT_EDGES_V
        n_out, edges = construct_simple(n, m)
        edges = edges[:m]
        familyname = f"{family}_{mode}"
        if not has_path(n_out, edges):
            raise ValueError(f"Generated {familyname}_{i} has no valid s->t path")
        _write_named_graph(folder, family, mode, i, n_out, edges)


def generate_simple_e():
    folder = "simple_e"
    family = "simple"
    mode = "e"
    for i in range(1, NUM_GRAPHS_PER_MODE + 1):
        random.seed(BASE_SEED + i)
        n = CONSTANT_VERTICES_E
        m = BASE_EDGES_E + (i - 1) * EDGE_INCREMENT
        n_out, edges = construct_simple(n, m)
        edges = edges[:m]
        if not has_path(n_out, edges):
            raise ValueError(f"Generated graph_{family}_{mode}_{i:02d} has no valid s->t path")
        _write_named_graph(folder, family, mode, i, n_out, edges)


def generate_worst_v():
    folder = "worst_v"
    family = "worst"
    mode = "v"
    for i in range(1, NUM_GRAPHS_PER_MODE + 1):
        random.seed(BASE_SEED + i)
        n = BASE_VERTICES_V + (i - 1) * VERTEX_INCREMENT
        m = CONSTANT_EDGES_V
        n_out, edges = construct_worst(n, m)
        edges = edges[:m]
        if not has_path(n_out, edges):
            raise ValueError(f"Generated graph_{family}_{mode}_{i:02d} has no valid s->t path")
        _write_named_graph(folder, family, mode, i, n_out, edges)


def generate_worst_e():
    folder = "worst_e"
    family = "worst"
    mode = "e"
    for i in range(1, NUM_GRAPHS_PER_MODE + 1):
        random.seed(BASE_SEED + i)
        n = CONSTANT_VERTICES_E
        m = BASE_EDGES_E + (i - 1) * EDGE_INCREMENT
        n_out, edges = construct_worst(n, m)
        edges = edges[:m]
        if not has_path(n_out, edges):
            raise ValueError(f"Generated graph_{family}_{mode}_{i:02d} has no valid s->t path")
        _write_named_graph(folder, family, mode, i, n_out, edges)


def main():
    """Generate all families and modes. This is a convenience wrapper."""
    print("Generating graphs for all families and modes...")
    generate_general_v()
    generate_general_e()
    generate_unitcap_v()
    generate_unitcap_e()
    generate_simple_v()
    generate_simple_e()
    generate_worst_v()
    generate_worst_e()
    print("Generation complete.")


if __name__ == "__main__":
    main()

