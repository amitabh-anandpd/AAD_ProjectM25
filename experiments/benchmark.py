import csv
import copy
import sys
import os
import time

# Add parent directory to path so we can import from algorithms/utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import importlib.util

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

FordFulkerson = import_from_path('fordfulkerson', './algorithms/fordfulkerson/code/ff.py')  # Update path if needed
Dinics = import_from_path('dinics', './algorithms/dinics/code/dinics.py')
PushRelabel = import_from_path('push_relabel', './algorithms/push_relabel/code/push_relabel.py')
BK = import_from_path('bk', './Boykov-Kolmogorov/working4.py')
from utils.graph_generator import GraphGenerator

def run_large_scale_benchmark():
    generator = GraphGenerator()
    results = []
    
    # --- CONFIGURATION ---
    # Testing with the sizes that caused the error + larger ones
    graph_sizes = [100,200,300,400] 
    trials = 3
    #families = ['sparse', 'dense', 'layered']
    families = ['sparse','layered','dense']

    print(f"{'Nodes':<6} | {'Family':<8} | {'Trial':<5} | {'FF (s)':<10} | {'Dinic (s)':<10} | {'PR (s)':<10} | {'Winner':<10}")
    print("-" * 85)

    for n in graph_sizes:
        for family in families:
            for t in range(trials):
                # 1. Generate Graph
                if family == 'sparse':
                    graph = generator.generate_sparse(n)
                elif family == 'dense':
                    graph = generator.generate_dense(n)
                elif family == 'layered':
                    graph = generator.generate_layered(n)
                
                # --- FIX START ---
                # Dynamic Sink Detection: 
                # Because the generator might renumber nodes to remove gaps, 
                # the highest node index is always len(graph) - 1.
                source = 0
                sink = len(graph) - 1 # <-- FIXED: Do not use 'n - 1'
                # --- FIX END ---

                def adjdict_to_edgelist(adj):
                    edges = []
                    for u, nbrs in adj.items():
                        for v, c in nbrs.items():
                            edges.append((u, v, c))
                    return edges

                num_vertices = len(graph)
                edge_list = adjdict_to_edgelist(graph)

                # 2. Run Algorithms
                # Deepcopy is ESSENTIAL because algorithms modify residual capacities in place.
                
                # --- Ford-Fulkerson ---
                graph_ff = copy.deepcopy(graph)
                start = time.perf_counter()
                ff = FordFulkerson.FordFulkerson(num_vertices, edge_list, source, sink)
                flow_ff, _ = ff.run()
                time_ff = time.perf_counter() - start

                # --- Dinic's ---
                graph_dn = copy.deepcopy(graph)
                dn = Dinics.Dinics(num_vertices, edge_list, source, sink)
                flow_dn, _ = dn.run()
                time_dn = time.perf_counter() - start

                # --- Push-Relabel ---
                # (Push-Relabel expects an adjacency dict, so your current usage is correct)
                graph_pr = copy.deepcopy(graph)
                pr = PushRelabel.PushRelabel(graph_pr, source, sink)
                flow_pr, time_pr = pr.run()

                # 3. Verification
                is_correct = (flow_ff == flow_dn == flow_pr)
                
                # 4. Determine Winner
                timings = {
                    'Ford-Fulkerson': time_ff, 
                    'Dinic': time_dn, 
                    'Push-Relabel': time_pr
                }
                winner = min(timings, key=timings.get)

                # 5. Log Results
                results.append({
                    'nodes': len(graph), # Log actual size
                    'family': family,
                    'trial': t,
                    'ff_time': time_ff,
                    'dinic_time': time_dn,
                    'pr_time': time_pr,
                    'winner': winner,
                    'max_flow': flow_ff,
                    'correct': is_correct
                })
                
                print(f"{len(graph):<6} | {family:<8} | {t:<5} | {time_ff:.6f}   | {time_dn:.6f}   | {time_pr:.6f}   | {winner:<10}")

    # 6. Save to CSV
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/final_benchmark.csv'
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nBenchmark Complete. Results saved to {csv_path}")

if __name__ == "__main__":
    run_large_scale_benchmark()