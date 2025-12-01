"""
Analysis module for Dinic's algorithm experimental results.

This module generates a summary table from performance data.

Usage:
    python3 code/analyze.py [batch_folder]
    
    If batch_folder is specified, analyzes only graphs from that batch folder.
    Valid batch folders: graphs, graphs_v, graphs_e
"""
import os
import sys
import argparse
import pandas as pd
from data_store import find_all_execution_data


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
        # Map graph_name -> folder
        mapping = {}
        for folder, path in exec_files:
            name = os.path.basename(path).replace("_execution_data.json", "")
            mapping[name] = folder
            mapping[name + ".txt"] = folder

        if mapping:
            df["graph_folder"] = df.get("graph_name", df.get("graph", df.index)).apply(lambda g: mapping.get(g, ""))
        else:
            # Cannot infer
            pass

    # Filter by batch folder if specified
    if batch_folder:
        if "graph_folder" in df.columns:
            df = df[df["graph_folder"] == batch_folder].copy()
        else:
            print(f"Warning: 'graph_folder' column not found and inference failed; cannot filter by batch.")
            return pd.DataFrame()
    
    # Convert numeric columns
    numeric_cols = ["n", "m", "total_time", "algorithm_time", "bfs_time_total", "dfs_time_total",
                    "num_iterations", "num_augmenting_paths", "max_flow", "min_cut_value"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Compute algorithm_time if not present (for backward compatibility)
    if "algorithm_time" not in df.columns:
        if "bfs_time_total" in df.columns and "dfs_time_total" in df.columns:
            df["algorithm_time"] = df["bfs_time_total"] + df["dfs_time_total"]
        else:
            df["algorithm_time"] = 0.0
    
    return df


def generate_summary_table(df: pd.DataFrame, output_path: str, batch_folder: str = None) -> None:
    """
    Generate a summary table from performance data.
    
    This function creates a clean, readable table with all performance metrics
    ordered by family and graph name.
    
    Args:
        df: DataFrame with performance data
        output_path: Path to save the table (CSV format)
    """
    if df.empty:
        print("No data to generate table from.")
        return
    
    # Ensure algorithm_time exists
    if "algorithm_time" not in df.columns:
        if "bfs_time_total" in df.columns and "dfs_time_total" in df.columns:
            df["algorithm_time"] = pd.to_numeric(df["bfs_time_total"], errors='coerce') + pd.to_numeric(df["dfs_time_total"], errors='coerce')
        else:
            df["algorithm_time"] = 0.0
    
    # Select and order columns
    columns = [
        "family", "graph_name", "n", "m", "total_time", "algorithm_time",
        "bfs_time_total", "dfs_time_total", "num_iterations",
        "num_augmenting_paths", "max_flow", "min_cut_value"
    ]
    
    # Filter to available columns (in order)
    available_columns = [col for col in columns if col in df.columns]
    table_df = df[available_columns].copy()
    
    # Sort by family, then graph name
    if "family" in table_df.columns:
        table_df = table_df.sort_values(["family", "graph_name"])
    
    # Format numeric columns for CSV (keep as numbers)
    # Save to CSV (with numeric values)
    table_df.to_csv(output_path, index=False)
    batch_label = f" for {batch_folder}" if batch_folder else ""
    print(f"Summary table saved to: {output_path}{batch_label}")
    
    # Create formatted display version
    display_df = table_df.copy()
    if "total_time" in display_df.columns:
        display_df["total_time"] = display_df["total_time"].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
    if "algorithm_time" in display_df.columns:
        display_df["algorithm_time"] = display_df["algorithm_time"].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
    if "bfs_time_total" in display_df.columns:
        display_df["bfs_time_total"] = display_df["bfs_time_total"].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
    if "dfs_time_total" in display_df.columns:
        display_df["dfs_time_total"] = display_df["dfs_time_total"].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
    
    # Print formatted tabular version with better formatting
    print("\n" + "="*160)
    print("PERFORMANCE SUMMARY TABLE".center(160))
    print("="*160)
    
    # Configure pandas display options for better table formatting
    with pd.option_context('display.max_columns', None,
                          'display.width', 200,
                          'display.max_colwidth', 25,
                          'display.float_format', lambda x: f'{x:.6f}' if pd.notna(x) else 'N/A'):
        print(display_df.to_string(index=False))
    
    print("="*160 + "\n")


def main():
    """
    Generate summary table from performance data.
    
    This is the main entry point for analysis. It loads performance data
    and generates a summary table, optionally filtered by batch folder.
    """
    parser = argparse.ArgumentParser(
        description="Generate summary table from performance data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 code/analyze.py graphs      # Analyze only graphs/ batch
  python3 code/analyze.py graphs_v    # Analyze only graphs_v/ batch
  python3 code/analyze.py graphs_e    # Analyze only graphs_e/ batch
  python3 code/analyze.py             # Analyze all batches
        """
    )
    parser.add_argument(
        "batch_folder",
        help="Batch folder to analyze (e.g., graphs, graphs_v, graphs_e, or any custom folder). This argument is required."
    )
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")
    
    df = load_performance_data(results_dir, args.batch_folder)
    
    if df.empty:
        if args.batch_folder:
            print(f"No performance data found for batch folder: {args.batch_folder}")
            print("Run experiments first using:")
            print(f"  python3 code/batch_run.py {args.batch_folder}")
        else:
            print("No performance data found. Run some experiments first using:")
            print("  python3 code/batch_run.py")
        return
    
    batch_label = f" ({args.batch_folder})" if args.batch_folder else ""
    print(f"Loaded {len(df)} performance record(s){batch_label}")
    
    # Generate summary table - store in batch-specific folder if specified
    if args.batch_folder:
        batch_results_dir = os.path.join(results_dir, args.batch_folder)
        os.makedirs(batch_results_dir, exist_ok=True)
        table_path = os.path.join(batch_results_dir, "summary_table.csv")
    else:
        table_path = os.path.join(results_dir, "summary_table.csv")
    
    generate_summary_table(df, table_path, args.batch_folder)


if __name__ == "__main__":
    main()
