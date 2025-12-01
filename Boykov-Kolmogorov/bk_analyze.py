import os
import pandas as pd
import matplotlib.pyplot as plt

def load_metrics(results_dir):
    # Directly load the central CSV file
    metrics_path = os.path.join(results_dir, "bk_all_metrics.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        return df
    else:
        print(f"No metrics file found at {metrics_path}")
        return pd.DataFrame()

def plot_runtime_vs_size(df, out_path):
    # For grid graphs, use the 'vertices' or 'input_name' column
    if 'vertices' in df.columns and 'runtime_sec' in df.columns:
        plt.figure(figsize=(8,6))
        plt.plot(df['vertices'], df['runtime_sec'], marker="o")
        plt.xlabel("Number of Vertices")
        plt.ylabel("Runtime (s)")
        plt.title("BK Runtime vs Number of Vertices")
        plt.savefig(out_path)
        plt.close()
    elif 'input_name' in df.columns and 'runtime_sec' in df.columns:
        # Try to extract size from input_name if possible
        try:
            df['size'] = df['input_name'].apply(lambda x: int(str(x).split('x')[0]))
            plt.figure(figsize=(8,6))
            plt.plot(df['size'], df['runtime_sec'], marker="o")
            plt.xlabel("Grid Size (N)")
            plt.ylabel("Runtime (s)")
            plt.title("BK Runtime vs Grid Size")
            plt.savefig(out_path)
            plt.close()
        except Exception as e:
            print("Could not extract size from input_name:", e)
    else:
        print("No suitable columns found for plotting.")

def main():
    results_dir = "Boykov-Kolmogorov/Boykov-Kolmogorov"
    df = load_metrics(results_dir)
    if not df.empty:
        plot_runtime_vs_size(df, "Boykov-Kolmogorov/results/bk_runtime_vs_size.png")
    else:
        print("No data to plot.")
    # ... more plots and markdown report generation

if __name__ == "__main__":
    main()