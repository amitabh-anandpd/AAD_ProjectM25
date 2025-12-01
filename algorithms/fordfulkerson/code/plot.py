"""
Plotting module for Ford–Fulkerson results (copied/adapted from Dinic's plot.py).

Usage:
  python3 code/plot.py <batch_folder>

This script reads `results/performance.csv` (method-specific results folders)
and generates scatter plots under `results/<batch_folder>/plots/`.
"""
import os
import sys
import argparse
import pandas as pd
from data_store import find_all_execution_data
import matplotlib.pyplot as plt
import numpy as np


def load_performance_data(results_dir: str, batch_folder: str = None) -> pd.DataFrame:
    csv_path = os.path.join(results_dir, "performance.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)

    if "graph_folder" not in df.columns:
        exec_files = find_all_execution_data(results_dir)
        mapping = {}
        for folder, path in exec_files:
            name = os.path.basename(path).replace("_execution_data.json", "")
            mapping[name] = folder
            mapping[name + ".txt"] = folder

        if mapping:
            df["graph_folder"] = df.get("graph_name", df.get("graph", df.index)).apply(lambda g: mapping.get(g, ""))

    if batch_folder:
        if "graph_folder" in df.columns:
            df = df[df["graph_folder"] == batch_folder].copy()
        else:
            print(f"Warning: 'graph_folder' column not found; cannot filter by {batch_folder}")
            return pd.DataFrame()

    numeric_cols = ["n", "m", "total_time", "bfs_time_total", "dfs_time_total",
                    "num_iterations", "num_augmenting_paths", "max_flow", "min_cut_value"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if "family" in df.columns:
        df["family"] = df["family"].fillna("").astype(str)

    return df


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, output_path: str, title: str = None) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        print(f"No data available for plot {y_col} vs {x_col}")
        return

    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df = df.dropna(subset=[x_col, y_col]).copy()
    if df.empty:
        print(f"No valid numeric data for plot {y_col} vs {x_col}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if "family" in df.columns:
        families = [f for f in df["family"].unique() if f and str(f).strip() != ""]
        if families:
            colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(families))))
            color_map = {fam: colors[i % len(colors)] for i, fam in enumerate(families)}
            for family in families:
                family_data = df[df["family"] == family]
                if not family_data.empty:
                    ax.scatter(
                        family_data[x_col],
                        family_data[y_col],
                        label=family,
                        color=color_map[family],
                        alpha=0.9,
                        s=40,
                        edgecolors='black',
                        linewidth=0.6
                    )
        else:
            ax.scatter(df[x_col], df[y_col], alpha=0.9, s=40, edgecolors='black', linewidth=0.6, color='steelblue')
    else:
        ax.scatter(df[x_col], df[y_col], alpha=0.7, s=100, edgecolors='black', linewidth=1, color='steelblue')

    xlabel = "Number of Vertices (n)" if x_col == "n" else ("Number of Edges (m)" if x_col == "m" else x_col)
    ylabel = "Total Time (seconds)" if y_col == "total_time" else ("Algorithm Time (seconds)" if y_col == "algorithm_time" else y_col)

    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title or f"{ylabel} vs {xlabel}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if "family" in df.columns and len([f for f in df["family"].unique() if f and str(f).strip() != ""]) > 0:
        ax.legend(loc='best', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate scatter plots from performance data")
    parser.add_argument("batch_folder", help="Batch folder to plot (required)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")

    plots_dir = os.path.join(results_dir, args.batch_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df = load_performance_data(results_dir, args.batch_folder)
    if df.empty:
        print(f"No performance data found for batch folder: {args.batch_folder}")
        return

    if "algorithm_time" not in df.columns:
        if "bfs_time_total" in df.columns and "dfs_time_total" in df.columns:
            df["algorithm_time"] = pd.to_numeric(df["bfs_time_total"], errors='coerce') + pd.to_numeric(df["dfs_time_total"], errors='coerce')
        else:
            df["algorithm_time"] = pd.to_numeric(df.get("total_time", pd.Series([0]*len(df))), errors='coerce')

    plot_scatter(df, "n", "algorithm_time", os.path.join(plots_dir, "scatter_algorithm_time_vs_vertices.png"), title="Algorithm Time vs Number of Vertices")
    plot_scatter(df, "m", "algorithm_time", os.path.join(plots_dir, "scatter_algorithm_time_vs_edges.png"), title="Algorithm Time vs Number of Edges")

    print(f"\nScatter plots generated successfully in: {plots_dir}")


if __name__ == "__main__":
    main()