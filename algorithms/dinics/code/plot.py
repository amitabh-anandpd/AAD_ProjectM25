"""
Plotting module for generating scatter plots from performance data.

This module creates scatter plots showing:
- Time vs number of edges (m)
- Time vs number of vertices (n)

Usage:
    python3 code/plot.py [batch_folder]
    
    If batch_folder is specified, generates plots only for graphs from that batch folder.
    Valid batch folders: graphs, graphs_v, graphs_e
"""
import os
import sys
import argparse
import pandas as pd
from data_store import find_all_execution_data
import matplotlib.pyplot as plt
import numpy as np


def load_performance_data(results_dir: str, batch_folder: str = None) -> pd.DataFrame:
    """
    Load performance.csv into a DataFrame, optionally filtered by batch folder.
    
    Args:
        results_dir: Directory containing performance.csv
        batch_folder: Optional batch folder name to filter by (graphs, graphs_v, graphs_e)
        
    Returns:
        DataFrame with performance data (filtered if batch_folder specified)
    """
    csv_path = os.path.join(results_dir, "performance.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    
    # If graph_folder column missing, try to infer from execution data locations
    if "graph_folder" not in df.columns:
        exec_files = find_all_execution_data(results_dir)
        mapping = {}
        for folder, path in exec_files:
            name = os.path.basename(path).replace("_execution_data.json", "")
            mapping[name] = folder
            mapping[name + ".txt"] = folder

        if mapping:
            df["graph_folder"] = df.get("graph_name", df.get("graph", df.index)).apply(lambda g: mapping.get(g, ""))
        else:
            pass

    # Filter by batch folder if specified
    if batch_folder:
        if "graph_folder" in df.columns:
            df = df[df["graph_folder"] == batch_folder].copy()
        else:
            print(f"Warning: 'graph_folder' column not found and inference failed; cannot filter by batch.")
            return pd.DataFrame()
    
    # Convert numeric columns
    numeric_cols = ["n", "m", "total_time", "bfs_time_total", "dfs_time_total",
                    "num_iterations", "num_augmenting_paths", "max_flow", "min_cut_value"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Normalize family column (avoid NaN or empty family labels)
    if "family" in df.columns:
        df["family"] = df["family"].fillna("").astype(str)
    
    return df


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, output_path: str, title: str = None) -> None:
    """Generic scatter plotter.

    Args:
        df: DataFrame with performance data
        x_col: Column name to use for x axis (e.g., 'n' or 'm')
        y_col: Column name to use for y axis (e.g., 'algorithm_time' or 'total_time')
        output_path: Path to save the plot
        title: Optional custom title
    """
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        print(f"No data available for plot {y_col} vs {x_col}")
        return

    # Ensure numeric and drop rows with NaN in plotted columns
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df = df.dropna(subset=[x_col, y_col]).copy()
    if df.empty:
        print(f"No valid numeric data for plot {y_col} vs {x_col} after cleaning")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by family for color coding
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
            # no meaningful family labels; plot all points
            ax.scatter(
                df[x_col],
                df[y_col],
                alpha=0.9,
                s=40,
                edgecolors='black',
                linewidth=0.6,
                color='steelblue'
            )
    else:
        # No family column, plot all points
        ax.scatter(
            df[x_col],
            df[y_col],
            alpha=0.7,
            s=100,
            edgecolors='black',
            linewidth=1,
            color='steelblue'
        )

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


def plot_time_vs_edges(df: pd.DataFrame, output_path: str) -> None:
    """
    Generate scatter plot: total_time vs number of edges (m).
    
    Args:
        df: DataFrame with performance data
        output_path: Path to save the plot
    """
    if df.empty or "m" not in df.columns or "total_time" not in df.columns:
        print("No data available for time vs edges plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by family for color coding
    if "family" in df.columns:
        families = [f for f in df["family"].unique() if pd.notna(f) and f != ""]
        colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
        color_map = {fam: colors[i] for i, fam in enumerate(families)}
        
        for family in families:
            family_data = df[df["family"] == family]
            if not family_data.empty:
                ax.scatter(
                    family_data["m"],
                    family_data["total_time"],
                    label=family,
                    color=color_map[family],
                    alpha=0.7,
                    s=100,
                    edgecolors='black',
                    linewidth=1
                )
    else:
        # No family column, plot all points
        ax.scatter(
            df["m"],
            df["total_time"],
            alpha=0.7,
            s=100,
            edgecolors='black',
            linewidth=1,
            color='steelblue'
        )
    
    ax.set_xlabel("Number of Edges (m)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Total Time (seconds)", fontsize=12, fontweight='bold')
    ax.set_title("Runtime vs Number of Edges", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if "family" in df.columns and len(families) > 0:
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """
    Generate scatter plots from performance data.
    
    This is run after analyze.py to create visual plots, optionally filtered by batch folder.
    """
    parser = argparse.ArgumentParser(
        description="Generate scatter plots from performance data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 code/plot.py graphs      # Plot only graphs/ batch
  python3 code/plot.py graphs_v    # Plot only graphs_v/ batch
  python3 code/plot.py graphs_e    # Plot only graphs_e/ batch
  python3 code/plot.py             # Plot all batches
        """
    )
    parser.add_argument(
        "batch_folder",
        help="Batch folder to plot (e.g., graphs, graphs_v, graphs_e, or any custom folder). This argument is required."
    )
    parser.add_argument(
        "-x",
        "--xcol",
        dest="xcol",
        help="Column to use for x-axis (e.g., 'n' or 'm'). If omitted, defaults to 'n' for first plot and 'm' for second.",
        default=None,
    )
    parser.add_argument(
        "-y",
        "--ycol",
        dest="ycol",
        help="Column to use for y-axis (e.g., 'algorithm_time' or 'total_time'). If omitted, defaults to 'algorithm_time'.",
        default=None,
    )
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")

    # Save plots under results: results/<batch>/plots or results/plots for global
    if args.batch_folder:
        plots_dir = os.path.join(results_dir, args.batch_folder, "plots")
    else:
        plots_dir = os.path.join(results_dir, "plots")

    os.makedirs(plots_dir, exist_ok=True)
    
    df = load_performance_data(results_dir, args.batch_folder)
    
    if df.empty:
        if args.batch_folder:
            print(f"No performance data found for batch folder: {args.batch_folder}")
            print("Run experiments first using:")
            print(f"  python3 code/batch_run.py {args.batch_folder}")
        else:
            print("No performance data found. Run experiments first using:")
            print("  python3 code/batch_run.py")
        return
    
    batch_label = f" ({args.batch_folder})" if args.batch_folder else ""
    print(f"Loaded {len(df)} performance record(s){batch_label}")
    
    # Ensure algorithm_time exists (compute if needed)
    if "algorithm_time" not in df.columns:
        if "bfs_time_total" in df.columns and "dfs_time_total" in df.columns:
            df["algorithm_time"] = pd.to_numeric(df["bfs_time_total"], errors='coerce') + pd.to_numeric(df["dfs_time_total"], errors='coerce')
        else:
            df["algorithm_time"] = pd.to_numeric(df.get("total_time", pd.Series([0]*len(df))), errors='coerce')

    # Generate scatter plots (default requested plots: algorithm_time vs vertices/edges)
    xcol = args.xcol
    ycol = args.ycol or "algorithm_time"

    if xcol:
        # Single custom plot requested
        out_name = f"scatter_{ycol}_vs_{xcol}.png"
        plot_scatter(df, xcol, ycol, os.path.join(plots_dir, out_name), title=f"{ycol} vs {xcol}")
    else:
        # Default two plots: algorithm_time vs vertices and algorithm_time vs edges
        plot_scatter(df, "n", ycol, os.path.join(plots_dir, "scatter_algorithm_time_vs_vertices.png"), title="Algorithm Time vs Number of Vertices")
        plot_scatter(df, "m", ycol, os.path.join(plots_dir, "scatter_algorithm_time_vs_edges.png"), title="Algorithm Time vs Number of Edges")
    
    print(f"\nScatter plots generated successfully in: {plots_dir}")


if __name__ == "__main__":
    main()

