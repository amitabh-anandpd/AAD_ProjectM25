"""
Analysis module for Ford–Fulkerson experimental results (adapted from Dinic).

Usage:
  python3 code/analyze.py <batch_folder>

Generates `results/<batch_folder>/summary_table.csv` and prints a formatted table.
"""
import os
import sys
import argparse
import pandas as pd
from data_store import find_all_execution_data


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

    numeric_cols = ["n", "m", "total_time", "algorithm_time", "bfs_time_total", "dfs_time_total",
                    "num_iterations", "num_augmenting_paths", "max_flow", "min_cut_value"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if "algorithm_time" not in df.columns:
        if "bfs_time_total" in df.columns and "dfs_time_total" in df.columns:
            df["algorithm_time"] = df["bfs_time_total"] + df["dfs_time_total"]
        else:
            df["algorithm_time"] = 0.0

    return df


def generate_summary_table(df: pd.DataFrame, output_path: str, batch_folder: str = None) -> None:
    if df.empty:
        print("No data to generate table from.")
        return

    if "algorithm_time" not in df.columns:
        if "bfs_time_total" in df.columns and "dfs_time_total" in df.columns:
            df["algorithm_time"] = pd.to_numeric(df["bfs_time_total"], errors='coerce') + pd.to_numeric(df["dfs_time_total"], errors='coerce')
        else:
            df["algorithm_time"] = 0.0

    columns = [
        "family", "graph_name", "n", "m", "total_time", "algorithm_time",
        "bfs_time_total", "dfs_time_total", "num_iterations",
        "num_augmenting_paths", "max_flow", "min_cut_value"
    ]

    available_columns = [col for col in columns if col in df.columns]
    table_df = df[available_columns].copy()
    if "family" in table_df.columns:
        table_df = table_df.sort_values(["family", "graph_name"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    table_df.to_csv(output_path, index=False)
    batch_label = f" for {batch_folder}" if batch_folder else ""
    print(f"Summary table saved to: {output_path}{batch_label}")

    display_df = table_df.copy()
    if "total_time" in display_df.columns:
        display_df["total_time"] = display_df["total_time"].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
    if "algorithm_time" in display_df.columns:
        display_df["algorithm_time"] = display_df["algorithm_time"].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
    if "bfs_time_total" in display_df.columns:
        display_df["bfs_time_total"] = display_df["bfs_time_total"].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
    if "dfs_time_total" in display_df.columns:
        display_df["dfs_time_total"] = display_df["dfs_time_total"].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")

    print("\n" + "="*160)
    print("PERFORMANCE SUMMARY TABLE".center(160))
    print("="*160)
    with pd.option_context('display.max_columns', None, 'display.width', 200, 'display.max_colwidth', 25, 'display.float_format', lambda x: f'{x:.6f}' if pd.notna(x) else 'N/A'):
        print(display_df.to_string(index=False))
    print("="*160 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate summary table from performance data")
    parser.add_argument("batch_folder", help="Batch folder to analyze (required)")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(__file__))
    results_dir = os.path.join(project_root, "results")

    df = load_performance_data(results_dir, args.batch_folder)
    if df.empty:
        print(f"No performance data found for batch folder: {args.batch_folder}")
        return

    if args.batch_folder:
        batch_results_dir = os.path.join(results_dir, args.batch_folder)
        os.makedirs(batch_results_dir, exist_ok=True)
        table_path = os.path.join(batch_results_dir, "summary_table.csv")
    else:
        table_path = os.path.join(results_dir, "summary_table.csv")

    generate_summary_table(df, table_path, args.batch_folder)


if __name__ == "__main__":
    main()