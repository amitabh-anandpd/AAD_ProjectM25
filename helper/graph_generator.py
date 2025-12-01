import random

def generate_strongly_connected_graph(n_nodes, n_edges,
                                      min_weight=1, max_weight=10):
    edges = set()          # stores (u, v)
    weighted_edges = []    # stores final (u, v, w)

    # --- STEP 1: Create Hamiltonian cycle to ensure strong connectivity
    nodes = list(range(n_nodes))
    random.shuffle(nodes)
    for i in range(n_nodes):
        u = nodes[i]
        v = nodes[(i + 1) % n_nodes]

        if (u, v) not in edges:
            w = random.randint(min_weight, max_weight)
            edges.add((u, v))
            weighted_edges.append((u, v, w))

    # --- STEP 2: Add remaining edges without duplicates
    remaining = max(0, n_edges - len(edges))

    while remaining > 0:
        u = random.randint(0, n_nodes - 1)
        v = random.randint(0, n_nodes - 1)

        if u == v:
            continue
        if (u, v) in edges:
            continue  # avoid duplicates (u,v) regardless of weight

        w = random.randint(min_weight, max_weight)
        edges.add((u, v))
        weighted_edges.append((u, v, w))
        remaining -= 1

    return weighted_edges


def save_graph(nodes, edges, filename="graph.txt"):
    with open(filename, "w") as f:
        f.write(f"{len(edges)} {nodes}\n")
        for u, v, w in edges:
            f.write(f"{u} {v} {w}\n")


# Example usage
if __name__ == "__main__":
    n = 10
    edges = generate_strongly_connected_graph(
        n_nodes=n,
        n_edges=25,
        min_weight=1,
        max_weight=30
    )
    save_graph(n, edges, "graph.txt")
    print("Generated strongly connected directed graph without duplicate edges.")
