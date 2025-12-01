"""
Final Optimized Graph Generator.

Fixes:
1. Performance: Pre-calculates edge pools for constant-N batches (fixes 'general_e' lag).
2. Cycles: Adds option to force DAGs (Directed Acyclic Graphs) to debug Dinic loops.
"""
import os
import random
import math
from typing import List, Tuple, Set
from collections import deque

# ---------------------------------------------------------------------------
# Global parameters
# ---------------------------------------------------------------------------
NUM_GRAPHS_PER_MODE = 200
BASE_VERTICES_V = 120
BASE_EDGES_E = 500
VERTEX_INCREMENT = 20
EDGE_INCREMENT = 200
CONSTANT_EDGES_V = 1000
CONSTANT_VERTICES_E = 500
BASE_SEED = 1000

# SET THIS TO TRUE if you want to prevent cycles (A->B->A)
# Useful if your Dinic algorithm is getting stuck in infinite recursion.
DAG_MODE = True

# ---------------------------------------------------------------------------
# Fast Pool Helpers
# ---------------------------------------------------------------------------

def get_all_possible_edges(n: int, is_dag: bool) -> List[Tuple[int, int]]:
    """Generates all possible pairs (u, v) efficiently."""
    pool = []
    if is_dag:
        # Only allow u -> v where u < v (Strict DAG)
        for u in range(n):
            for v in range(u + 1, n):
                pool.append((u, v))
    else:
        # Allow cycles, just no self-loops
        for u in range(n):
            for v in range(n):
                if u != v:
                    pool.append((u, v))
    return pool

def get_edges_from_pool(n: int, current_edges: List[Tuple[int, int, int]], 
                        target_total: int, capacity_fn, 
                        precomputed_pool: List[Tuple[int, int]] = None) -> List[Tuple[int, int, int]]:
    """
    Super-fast sampling using list comprehension and set subtraction.
    """
    if len(current_edges) >= target_total:
        return current_edges

    needed = target_total - len(current_edges)
    existing_set = set((u, v) for u, v, _ in current_edges)

    # Use precomputed pool if available (fastest), otherwise generate on fly
    source_pool = precomputed_pool if precomputed_pool else get_all_possible_edges(n, DAG_MODE)

    # Fast filtering: Python list comprehension is optimized in C
    available = [pair for pair in source_pool if pair not in existing_set]

    if len(available) < needed:
        needed = len(available)

    chosen_pairs = random.sample(available, needed)
    
    new_edges = []
    for u, v in chosen_pairs:
        new_edges.append((u, v, capacity_fn()))
        
    return current_edges + new_edges

# ---------------------------------------------------------------------------
# Construction Logic
# ---------------------------------------------------------------------------

def construct_general(n: int, m: int, precomputed_pool=None) -> Tuple[int, List[Tuple[int, int, int]]]:
    edges = []
    
    # 1. Guarantee Path
    # We create a random path from 0 -> ... -> n-1
    intermediate = list(range(1, n - 1))
    random.shuffle(intermediate)
    
    # Pick a random path length
    path_len = random.randint(1, min(len(intermediate), max(2, int(n * 0.2))))
    path_nodes = [0] + intermediate[:path_len] + [n - 1]
    
    # If DAG_MODE is on, we must sort the path nodes to ensure u < v
    if DAG_MODE:
        path_nodes.sort()
        # Ensure 0 is first and n-1 is last (sort handles this naturally as 0 is min, n-1 is max)

    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i+1]
        edges.append((u, v, random.randint(1, 100)))

    # 2. Fill remaining
    return n, get_edges_from_pool(n, edges, m, lambda: random.randint(1, 100), precomputed_pool)

# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_graph(num_vertices: int, edges: List[Tuple[int, int, int]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"{num_vertices} {len(edges)}\n")
        for u, v, c in edges:
            f.write(f"{u} {v} {c}\n")

def _write_named_graph(folder: str, family: str, mode: str, i: int, n: int, edges: List[Tuple[int, int, int]]):
    # Saves to project_root/folder/filename
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, folder)
    filename = f"{family}_{mode}_{i}.txt"
    save_graph(n, edges, os.path.join(out_dir, filename))

# ---------------------------------------------------------------------------
# Main Generators
# ---------------------------------------------------------------------------

def generate_general_batches():
    # --- Batch 1: Varying Vertices (Cannot precompute pool efficiently) ---
    print(f"--- Generating general_v (DAG_MODE={DAG_MODE}) ---")
    for i in range(1, NUM_GRAPHS_PER_MODE + 1):
        random.seed(BASE_SEED + i)
        n = BASE_VERTICES_V + (i - 1) * VERTEX_INCREMENT
        m = CONSTANT_EDGES_V
        
        # Clip m to max possible edges
        max_edges = (n * (n - 1)) // 2 if DAG_MODE else n * (n - 1)
        m = min(m, max_edges)

        n_out, edges = construct_general(n, m)
        _write_named_graph("general_v", "general", "v", i, n_out, edges)
        
        if i % 50 == 0: print(f"  v-mode: {i}/{NUM_GRAPHS_PER_MODE}")

    # --- Batch 2: Varying Edges (Constant Vertices - Optimized!) ---
    print(f"--- Generating general_e (DAG_MODE={DAG_MODE}) ---")
    
    # PRE-CALCULATE POOL ONCE (Huge Speedup)
    fixed_n = CONSTANT_VERTICES_E
    print(f"  Pre-calculating edge pool for N={fixed_n}...")
    shared_pool = get_all_possible_edges(fixed_n, DAG_MODE)
    print(f"  Pool ready size={len(shared_pool)}. Starting generation...")

    for i in range(1, NUM_GRAPHS_PER_MODE + 1):
        random.seed(BASE_SEED + i)
        n = fixed_n
        m = BASE_EDGES_E + (i - 1) * EDGE_INCREMENT
        
        # Clip m
        max_edges = len(shared_pool)
        if m > max_edges: m = max_edges
        
        # Pass the shared pool to avoid rebuilding it 200 times
        n_out, edges = construct_general(n, m, precomputed_pool=shared_pool)
        _write_named_graph("general_e", "general", "e", i, n_out, edges)
        
        if i % 50 == 0: print(f"  e-mode: {i}/{NUM_GRAPHS_PER_MODE}")

def main():
    generate_general_batches()
    print("Generation complete.")

if __name__ == "__main__":
    main()