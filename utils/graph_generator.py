import random
import networkx as nx

class GraphGenerator:
    """
    Generates synthetic flow networks in standard adjacency dictionary format:
    {u: {v: capacity, ...}, ...}
    """
    
    def _to_adj_dict(self, G, capacity_range=(1, 100)):
        """Converts NetworkX graph to standard dict format with random capacities."""
        # CRITICAL FIX: Renumber nodes to 0..N-1 to ensure contiguous indices
        # This prevents IndexError in list-based algorithms if G has gaps in node IDs
        G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='sorted')
        
        adj = {i: {} for i in range(len(G.nodes))}
        for u, v in G.edges():
            adj[u][v] = random.randint(*capacity_range)
        return adj

    def generate_sparse(self, num_nodes):
        """Generates a sparse graph (Edges ≈ Nodes). Good for FF."""
        # Path graph ensures minimal connectivity 0->1->...->N-1
        G = nx.path_graph(num_nodes, create_using=nx.DiGraph)
        
        # Add 20% extra random edges to create some cycles/shortcuts
        num_extra = int(num_nodes * 0.2)
        # Safety check for very small graphs
        if num_nodes > 2:
            for _ in range(num_extra):
                u = random.randint(0, num_nodes - 2)
                v = random.randint(u + 1, num_nodes - 1)
                if u != v and not G.has_edge(u, v):
                    G.add_edge(u, v)
        return self._to_adj_dict(G)

    def generate_dense(self, num_nodes):
        """Generates a dense graph (Edges ≈ 0.5 * Nodes^2). Good for Dinic/PR."""
        G = nx.gnp_random_graph(num_nodes, 0.5, directed=True)
        # Guarantee s->t path exists
        if not nx.has_path(G, 0, num_nodes - 1):
            G.add_edge(0, num_nodes - 1)
        return self._to_adj_dict(G)

    def generate_layered(self, num_nodes, layers=4):
        """Generates a layered graph (typical for Dinic's worst/best cases)."""
        G = nx.DiGraph()
        # Determine nodes per layer; handle cases where num_nodes < layers
        effective_layers = min(layers, num_nodes)
        nodes_per_layer = max(1, num_nodes // effective_layers)
        
        # Create edges between layers
        for i in range(effective_layers - 1):
            # Current layer range
            u_start = i * nodes_per_layer
            u_end = (i + 1) * nodes_per_layer
            
            # Next layer range
            v_start = u_end
            v_end = (i + 2) * nodes_per_layer
            
            for u in range(u_start, u_end):
                for v in range(v_start, v_end):
                    # 50% chance to connect
                    if random.random() > 0.5:
                        G.add_edge(u, v)
                        
        # Ensure graph is connected end-to-end manually to avoid disconnected components
        # Connect Source (0) to the start of the chain
        # Connect end of chain to Sink (num_nodes-1)
        if not G.has_edge(0, 1):
            G.add_edge(0, 1)
        
        # Ensure the last generated node connects to the true sink index
        max_generated_node = max(G.nodes) if G.nodes else 0
        if max_generated_node < num_nodes - 1:
             G.add_edge(max_generated_node, num_nodes - 1)
             
        return self._to_adj_dict(G)