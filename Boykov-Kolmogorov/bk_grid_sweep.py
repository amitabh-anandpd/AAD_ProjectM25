import os
import subprocess
import pandas as pd

OUTPUT_ROOT = "./Boykov-Kolmogorov/bk_grid_sweep_results"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

all_metrics = []

for size in range(1, 129):
    print(f"Running BK for grid size {size}x{size}...")
    subprocess.run([
        "python", "bk_runner.py",
        "--size", str(size),
        "--no_folders"
    ], cwd=os.path.dirname(__file__))
print("Sweep complete. Results in: Boykov-Kolmogorov/bk_all_metrics.csv")
