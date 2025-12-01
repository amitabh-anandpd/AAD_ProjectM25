"""
Analysis and plotting tools for Push-Relabel algorithm experimental results.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from typing import Dict


def load_performance_data(results_dir: str) -> pd.DataFrame:
    """Load performance.csv into a DataFrame."""
    csv_path = os.path.join(results_dir, "performance.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    numeric_cols = ["n", "m", "seed", "trial", "max_flow", "total_time", 
                    "num_pushes", "num_relabels", "num_operations"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def plot_time_vs_n(df: pd.DataFrame, output_path: str) -> None:
    """Plot runtime vs number of vertices."""
    if df.empty or "family" not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    families = [f for f in df["family"].unique() if pd.notna(f) and f != ""]
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    
    for i, family in enumerate(families):
        family_data = df[df["family"] == family]
        if family_data.empty:
            continue
        
        grouped = family_data.groupby("n")["total_time"].agg(["mean", "std", "count"])
        grouped = grouped[grouped["count"] > 0]
        
        if len(grouped) > 0:
            x = grouped.index
            y_mean = grouped["mean"]
            y_std = grouped["std"].fillna(0)  # Replace NaN with 0 for single data points
            
            ax.errorbar(x, y_mean, yerr=y_std, label=family, color=colors[i], 
                       marker='o', linestyle='-', linewidth=2, markersize=8, capsize=5)
    
    ax.set_xlabel("Number of Vertices (n)", fontsize=12)
    ax.set_ylabel("Total Time (seconds)", fontsize=12)
    ax.set_title("Push-Relabel: Runtime vs Graph Size", fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_operations_vs_n(df: pd.DataFrame, output_path: str) -> None:
    """Plot number of operations vs graph size."""
    if df.empty or "num_operations" not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    families = [f for f in df["family"].unique() if pd.notna(f) and f != ""]
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    
    for i, family in enumerate(families):
        family_data = df[df["family"] == family]
        if family_data.empty:
            continue
        
        grouped = family_data.groupby("n")["num_operations"].agg(["mean", "std"])
        grouped = grouped[grouped["mean"] > 0]
        
        if len(grouped) > 0:
            x = grouped.index
            y_mean = grouped["mean"]
            y_std = grouped["std"].fillna(0)  # Replace NaN with 0 for single data points
            
            ax.errorbar(x, y_mean, yerr=y_std, label=family, color=colors[i],
                       marker='s', linestyle='--', linewidth=2, markersize=8, capsize=5)
    
    ax.set_xlabel("Number of Vertices (n)", fontsize=12)
    ax.set_ylabel("Total Operations (Push + Relabel)", fontsize=12)
    ax.set_title("Push-Relabel: Operations vs Graph Size", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_push_relabel_ratio(df: pd.DataFrame, output_path: str) -> None:
    """Plot ratio of push to relabel operations."""
    if df.empty or "num_pushes" not in df.columns or "num_relabels" not in df.columns:
        return
    
    df = df.copy()
    df["push_relabel_ratio"] = df["num_pushes"] / df["num_relabels"].replace(0, 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    families = [f for f in df["family"].unique() if pd.notna(f) and f != ""]
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    
    for i, family in enumerate(families):
        family_data = df[df["family"] == family]
        if family_data.empty:
            continue
        
        grouped = family_data.groupby("n")["push_relabel_ratio"].agg(["mean", "std"])
        grouped = grouped[grouped["mean"] > 0]
        
        if len(grouped) > 0:
            x = grouped.index
            y_mean = grouped["mean"]
            y_std = grouped["std"].fillna(0)  # Replace NaN with 0 for single data points
            
            ax.errorbar(x, y_mean, yerr=y_std, label=family, color=colors[i],
                       marker='^', linestyle='-.', linewidth=2, markersize=8, capsize=5)
    
    ax.set_xlabel("Number of Vertices (n)", fontsize=12)
    ax.set_ylabel("Push/Relabel Ratio", fontsize=12)
    ax.set_title("Push-Relabel: Operation Ratio Analysis", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_plot(df: pd.DataFrame, output_path: str) -> None:
    """Generate comparative summary plots."""
    if df.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    group_col = "graph_filename" if "graph_filename" in df.columns else "family"
    grouped = df.groupby(group_col).agg({
        "max_flow": "mean",
        "num_pushes": "mean",
        "num_relabels": "mean",
        "total_time": "mean"
    }).reset_index()
    
    x_pos = np.arange(len(grouped))
    
    # Plot 1: Max Flow
    axes[0, 0].bar(x_pos, grouped["max_flow"], color='steelblue', alpha=0.7)
    axes[0, 0].set_xlabel("Graph", fontsize=10)
    axes[0, 0].set_ylabel("Max Flow", fontsize=10)
    axes[0, 0].set_title("Maximum Flow", fontsize=11, fontweight='bold')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(grouped[group_col], rotation=45, ha='right', fontsize=8)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Push Operations
    axes[0, 1].bar(x_pos, grouped["num_pushes"], color='coral', alpha=0.7)
    axes[0, 1].set_xlabel("Graph", fontsize=10)
    axes[0, 1].set_ylabel("Push Operations", fontsize=10)
    axes[0, 1].set_title("Number of Push Operations", fontsize=11, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(grouped[group_col], rotation=45, ha='right', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Relabel Operations
    axes[1, 0].bar(x_pos, grouped["num_relabels"], color='mediumseagreen', alpha=0.7)
    axes[1, 0].set_xlabel("Graph", fontsize=10)
    axes[1, 0].set_ylabel("Relabel Operations", fontsize=10)
    axes[1, 0].set_title("Number of Relabel Operations", fontsize=11, fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(grouped[group_col], rotation=45, ha='right', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Runtime
    axes[1, 1].bar(x_pos, grouped["total_time"], color='orchid', alpha=0.7)
    axes[1, 1].set_xlabel("Graph", fontsize=10)
    axes[1, 1].set_ylabel("Runtime (seconds)", fontsize=10)
    axes[1, 1].set_title("Total Runtime", fontsize=11, fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(grouped[group_col], rotation=45, ha='right', fontsize=8)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def fit_scaling_laws(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Fit empirical scaling laws."""
    results = {}
    
    if df.empty:
        return results
    
    families = [f for f in df["family"].unique() if pd.notna(f) and f != ""]
    
    for family in families:
        family_data = df[df["family"] == family]
        if len(family_data) < 3:
            continue
        
        family_results = {}
        
        # Fit time vs n
        if "n" in family_data.columns and "total_time" in family_data.columns:
            x = family_data["n"].values
            y = family_data["total_time"].values
            x_log = np.log(x[x > 0])
            y_log = np.log(y[y > 0])
            
            if len(x_log) == len(y_log) and len(x_log) >= 2:
                slope, intercept, r_value, _, _ = stats.linregress(x_log, y_log)
                family_results["n_exponent"] = slope
                family_results["n_coefficient"] = np.exp(intercept)
                family_results["n_r_squared"] = r_value ** 2
        
        # Fit time vs m
        if "m" in family_data.columns and "total_time" in family_data.columns:
            x = family_data["m"].values
            y = family_data["total_time"].values
            x_log = np.log(x[x > 0])
            y_log = np.log(y[y > 0])
            
            if len(x_log) == len(y_log) and len(x_log) >= 2:
                slope, intercept, r_value, _, _ = stats.linregress(x_log, y_log)
                family_results["m_exponent"] = slope
                family_results["m_coefficient"] = np.exp(intercept)
                family_results["m_r_squared"] = r_value ** 2
        
        if family_results:
            results[family] = family_results
    
    return results


def main():
    """Generate all analysis plots."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")
    visuals_dir = os.path.join(project_root, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)
    
    df = load_performance_data(results_dir)
    
    if df.empty:
        print("No performance data found. Run experiments first.")
        return
    
    print(f"Loaded {len(df)} performance records")
    
    # Generate plots
    plot_time_vs_n(df, os.path.join(visuals_dir, "plot_time_vs_n.png"))
    plot_operations_vs_n(df, os.path.join(visuals_dir, "plot_operations_vs_n.png"))
    plot_push_relabel_ratio(df, os.path.join(visuals_dir, "plot_push_relabel_ratio.png"))
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
