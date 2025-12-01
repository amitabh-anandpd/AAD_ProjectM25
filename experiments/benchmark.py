import csv
import copy
import sys
import os
import time

# Add parent directory to path so we can import from algorithms/utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.graph_generator import GraphGenerator
from algorithms.fordflurkson import FordFulkerson
from algorithms.dinics import Dinics
from algorithms.push_relabel import PushRelabel

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

                # 2. Run Algorithms
                # Deepcopy is ESSENTIAL because algorithms modify residual capacities in place.
                
                # --- Ford-Fulkerson ---
                graph_ff = copy.deepcopy(graph)
                start = time.perf_counter()
                ff = FordFulkerson(graph_ff, source, sink)
                flow_ff = ff.run()
                time_ff = time.perf_counter() - start

                # --- Dinic's ---
                graph_dn = copy.deepcopy(graph)
                dn = Dinics(graph_dn, source, sink)
                flow_dn, time_dn = dn.run()

                # --- Push-Relabel ---
                graph_pr = copy.deepcopy(graph)
                pr = PushRelabel(graph_pr, source, sink)
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