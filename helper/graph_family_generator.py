import os
import random
import math
from pathlib import Path

# Configuration
NUM_GRAPHS = 300
V_START = 100
V_INCREMENT = 20
BASE_SEED = 42
MIN_CAPACITY = 1
MAX_CAPACITY = 100

def ensure_path(n, edges_set, rng):
    """Ensure at least one path from source (0) to sink (n-1)"""
    if n <= 1:
        return []
    
    # Create a random path from 0 to n-1
    intermediate = list(range(1, n-1))
    rng.shuffle(intermediate)
    
    path_nodes = [0] + intermediate + [n-1]
    
    # Add edges along this path with capacities
    path_edges = []
    for i in range(len(path_nodes) - 1):
        u, v = path_nodes[i], path_nodes[i+1]
        cap = rng.randint(MIN_CAPACITY, MAX_CAPACITY)
        path_edges.append((u, v, cap))  # Note: 3-tuple with capacity
        edges_set.add((u, v))
    
    return path_edges

def generate_sparse(n, i):
    """Generate sparse graph with m ≈ 2n edges"""
    seed = BASE_SEED + i
    rng = random.Random(seed)
    
    m = n * 2
    edges_set = set()
    
    # Ensure s-t path
    edges = ensure_path(n, edges_set, rng)
    
    # Add remaining random edges
    while len(edges) < m:
        u = rng.randint(0, n-1)
        v = rng.randint(0, n-1)
        if u != v and (u, v) not in edges_set:
            cap = rng.randint(MIN_CAPACITY, MAX_CAPACITY)
            edges.append((u, v, cap))
            edges_set.add((u, v))
    
    return n, edges

def generate_dense(n, i):
    """Generate dense graph with m ≈ n²/4 edges"""
    seed = BASE_SEED + i
    rng = random.Random(seed)
    
    m = (n * (n - 1)) // 4
    edges_set = set()
    
    # Ensure s-t path
    edges = ensure_path(n, edges_set, rng)
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(n) if u != v and (u, v) not in edges_set]
    
    # Sample remaining edges
    needed = m - len(edges)
    sampled = rng.sample(all_edges, min(needed, len(all_edges)))
    
    for u, v in sampled:
        cap = rng.randint(MIN_CAPACITY, MAX_CAPACITY)
        edges.append((u, v, cap))
    
    return n, edges

def generate_complete(n, i):
    """Generate complete graph with all possible directed edges"""
    seed = BASE_SEED + i
    rng = random.Random(seed)
    
    edges = []
    
    # Generate all pairs (u, v) where u != v
    for u in range(n):
        for v in range(n):
            if u != v:
                cap = rng.randint(MIN_CAPACITY, MAX_CAPACITY)
                edges.append((u, v, cap))
    
    return n, edges

def generate_dag(n, i):
    """Generate acyclic graph (DAG) with m ≈ n*log(n) edges"""
    seed = BASE_SEED + i
    rng = random.Random(seed)
    
    m = int(n * math.log(n))
    edges_set = set()
    
    # Ensure s-t path (automatically satisfies u < v)
    edges = ensure_path(n, edges_set, rng)
    
    # Precompute all valid DAG edges (u < v)
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edges_set]
    
    # Sample remaining edges
    needed = m - len(edges)
    sampled = rng.sample(all_edges, min(needed, len(all_edges)))
    
    for u, v in sampled:
        cap = rng.randint(MIN_CAPACITY, MAX_CAPACITY)
        edges.append((u, v, cap))
    
    return n, edges

def save_graph(filepath, n, edges):
    """Save graph to file in the specified format"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write(f"{n} {len(edges)}\n")
        for u, v, cap in edges:  # Unpacking 3-tuple
            f.write(f"{u} {v} {cap}\n")

def generate_all_graphs():
    """Generate all four families of graphs"""
    base_dir = Path("graphs")
    
    families = {
        "sparse": generate_sparse,
        "dense": generate_dense,
        "complete": generate_complete,
        "dag": generate_dag
    }
    
    for family_name, generator_func in families.items():
        print(f"Generating {family_name} graphs...")
        family_dir = base_dir / family_name
        
        for i in range(1, NUM_GRAPHS + 1):
            n = V_START + (i - 1) * V_INCREMENT
            
            # Generate graph
            n, edges = generator_func(n, i)
            
            # Save to file
            filename = f"{family_name}_{i:03d}.txt"
            filepath = family_dir / filename
            save_graph(filepath, n, edges)
            
            if i % 50 == 0:
                print(f"  Generated {i}/{NUM_GRAPHS} graphs")
        
        print(f"  Completed {family_name} family!\n")
    
    print("All graphs generated successfully!")

if __name__ == "__main__":
    generate_all_graphs()