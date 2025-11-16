"""
Analysis and plotting tools for Dinic's algorithm experimental results.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


def load_performance_data(results_dir: str) -> pd.DataFrame:
    """Load performance.csv into a DataFrame."""
    csv_path = os.path.join(results_dir, "performance.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # Convert numeric columns
    numeric_cols = ["n", "m", "seed", "trial", "max_flow", "total_time", "bfs_time_total", 
                    "dfs_time_total", "num_iterations", "num_augmenting_paths"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def plot_time_vs_n(df: pd.DataFrame, output_path: str) -> None:
    """Plot 1: total_time vs n for each family (log-log)."""
    if df.empty or "family" not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    families = [f for f in df["family"].unique() if pd.notna(f) and f != ""]
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    
    for i, family in enumerate(families):
        family_data = df[df["family"] == family]
        if family_data.empty:
            continue
        
        # Group by n and compute mean/std
        grouped = family_data.groupby("n")["total_time"].agg(["mean", "std", "count"])
        grouped = grouped[grouped["count"] > 0]
        
        if len(grouped) > 0:
            x = grouped.index
            y_mean = grouped["mean"]
            y_std = grouped["std"]
            
            ax.errorbar(x, y_mean, yerr=y_std, label=family, color=colors[i], 
                       marker='o', linestyle='-', linewidth=2, markersize=8, capsize=5)
    
    ax.set_xlabel("Number of Vertices (n)", fontsize=12)
    ax.set_ylabel("Total Time (seconds)", fontsize=12)
    ax.set_title("Runtime vs Graph Size (by Family)", fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_time_vs_m(df: pd.DataFrame, output_path: str) -> None:
    """Plot 2: total_time vs m (edges), log-log."""
    if df.empty or "m" not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    families = [f for f in df["family"].unique() if pd.notna(f) and f != ""]
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    
    for i, family in enumerate(families):
        family_data = df[df["family"] == family]
        if family_data.empty:
            continue
        
        grouped = family_data.groupby("m")["total_time"].agg(["mean", "std"])
        grouped = grouped[grouped["mean"] > 0]
        
        if len(grouped) > 0:
            x = grouped.index
            y_mean = grouped["mean"]
            y_std = grouped["std"]
            
            ax.errorbar(x, y_mean, yerr=y_std, label=family, color=colors[i],
                       marker='s', linestyle='--', linewidth=2, markersize=8, capsize=5)
    
    ax.set_xlabel("Number of Edges (m)", fontsize=12)
    ax.set_ylabel("Total Time (seconds)", fontsize=12)
    ax.set_title("Runtime vs Number of Edges (by Family)", fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_iterations_vs_n(df: pd.DataFrame, output_path: str) -> None:
    """Plot 3: num_iterations vs n (to show algorithmic behavior)."""
    if df.empty or "num_iterations" not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    families = [f for f in df["family"].unique() if pd.notna(f) and f != ""]
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    
    for i, family in enumerate(families):
        family_data = df[df["family"] == family]
        if family_data.empty:
            continue
        
        grouped = family_data.groupby("n")["num_iterations"].agg(["mean", "std"])
        grouped = grouped[grouped["mean"] > 0]
        
        if len(grouped) > 0:
            x = grouped.index
            y_mean = grouped["mean"]
            y_std = grouped["std"]
            
            ax.errorbar(x, y_mean, yerr=y_std, label=family, color=colors[i],
                       marker='^', linestyle='-.', linewidth=2, markersize=8, capsize=5)
    
    ax.set_xlabel("Number of Vertices (n)", fontsize=12)
    ax.set_ylabel("Number of BFS Phases (Iterations)", fontsize=12)
    ax.set_title("BFS Phases vs Graph Size (by Family)", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_time_per_path(df: pd.DataFrame, output_path: str) -> None:
    """Plot 4: avg time per augmenting path to show efficiency."""
    if df.empty or "num_augmenting_paths" not in df.columns:
        return
    
    # Compute time per path
    df = df.copy()
    df["time_per_path"] = df["total_time"] / df["num_augmenting_paths"].replace(0, 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    families = [f for f in df["family"].unique() if pd.notna(f) and f != ""]
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    
    for i, family in enumerate(families):
        family_data = df[df["family"] == family]
        if family_data.empty:
            continue
        
        grouped = family_data.groupby("n")["time_per_path"].agg(["mean", "std"])
        grouped = grouped[grouped["mean"] > 0]
        
        if len(grouped) > 0:
            x = grouped.index
            y_mean = grouped["mean"]
            y_std = grouped["std"]
            
            ax.errorbar(x, y_mean, yerr=y_std, label=family, color=colors[i],
                       marker='d', linestyle=':', linewidth=2, markersize=8, capsize=5)
    
    ax.set_xlabel("Number of Vertices (n)", fontsize=12)
    ax.set_ylabel("Time per Augmenting Path (seconds)", fontsize=12)
    ax.set_title("Efficiency: Time per Path vs Graph Size", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def fit_scaling_laws(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Fit empirical scaling laws: time ≈ a * n^b or time ≈ a * m^c."""
    results = {}
    
    if df.empty:
        return results
    
    families = [f for f in df["family"].unique() if pd.notna(f) and f != ""]
    
    for family in families:
        family_data = df[df["family"] == family]
        if len(family_data) < 3:
            continue
        
        family_results = {}
        
        # Fit time vs n: time = a * n^b
        if "n" in family_data.columns and "total_time" in family_data.columns:
            x = family_data["n"].values
            y = family_data["total_time"].values
            x_log = np.log(x[x > 0])
            y_log = np.log(y[y > 0])
            
            if len(x_log) == len(y_log) and len(x_log) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
                family_results["n_exponent"] = slope
                family_results["n_coefficient"] = np.exp(intercept)
                family_results["n_r_squared"] = r_value ** 2
        
        # Fit time vs m: time = a * m^c
        if "m" in family_data.columns and "total_time" in family_data.columns:
            x = family_data["m"].values
            y = family_data["total_time"].values
            x_log = np.log(x[x > 0])
            y_log = np.log(y[y > 0])
            
            if len(x_log) == len(y_log) and len(x_log) >= 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
                family_results["m_exponent"] = slope
                family_results["m_coefficient"] = np.exp(intercept)
                family_results["m_r_squared"] = r_value ** 2
        
        if family_results:
            results[family] = family_results
    
    return results


def generate_summary_plot(df: pd.DataFrame, output_path: str) -> None:
    """Generate comparative summary plot: total_flow, iterations, runtime vs sample_name."""
    if df.empty:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Group by graph_filename (for samples) or family
    if "graph_filename" in df.columns:
        group_col = "graph_filename"
    elif "family" in df.columns:
        group_col = "family"
    else:
        return
    
    grouped = df.groupby(group_col).agg({
        "max_flow": "mean",
        "num_iterations": "mean",
        "total_time": "mean"
    }).reset_index()
    
    x_pos = np.arange(len(grouped))
    
    # Plot 1: Max Flow
    axes[0].bar(x_pos, grouped["max_flow"], color='steelblue', alpha=0.7)
    axes[0].set_xlabel("Graph", fontsize=11)
    axes[0].set_ylabel("Max Flow", fontsize=11)
    axes[0].set_title("Maximum Flow", fontsize=12, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(grouped[group_col], rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Iterations
    axes[1].bar(x_pos, grouped["num_iterations"], color='coral', alpha=0.7)
    axes[1].set_xlabel("Graph", fontsize=11)
    axes[1].set_ylabel("BFS Phases", fontsize=11)
    axes[1].set_title("Number of Iterations", fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(grouped[group_col], rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Runtime
    axes[2].bar(x_pos, grouped["total_time"], color='mediumseagreen', alpha=0.7)
    axes[2].set_xlabel("Graph", fontsize=11)
    axes[2].set_ylabel("Runtime (seconds)", fontsize=11)
    axes[2].set_title("Total Runtime", fontsize=12, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(grouped[group_col], rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all analysis plots and report."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")
    visuals_dir = os.path.join(project_root, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)
    
    df = load_performance_data(results_dir)
    
    if df.empty:
        print("No performance data found. Run some experiments first.")
        return
    
    print(f"Loaded {len(df)} performance records")
    
    # Generate all plots
    plot_time_vs_n(df, os.path.join(visuals_dir, "plot_time_vs_n.png"))
    plot_time_vs_m(df, os.path.join(visuals_dir, "plot_time_vs_m.png"))
    plot_iterations_vs_n(df, os.path.join(visuals_dir, "plot_iterations_vs_n.png"))
    plot_time_per_path(df, os.path.join(visuals_dir, "plot_time_per_path.png"))
    generate_summary_plot(df, os.path.join(visuals_dir, "summary.png"))
    
    # Fit scaling laws
    scaling_laws = fit_scaling_laws(df)
    if scaling_laws:
        print("\n=== Scaling Law Fits ===")
        for family, results in scaling_laws.items():
            print(f"\n{family}:")
            if "n_exponent" in results:
                print(f"  Time vs n: {results['n_coefficient']:.4e} * n^{results['n_exponent']:.2f} (R²={results['n_r_squared']:.3f})")
            if "m_exponent" in results:
                print(f"  Time vs m: {results['m_coefficient']:.4e} * m^{results['m_exponent']:.2f} (R²={results['m_r_squared']:.3f})")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()

