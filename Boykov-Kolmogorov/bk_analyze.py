import os
import pandas as pd
import matplotlib.pyplot as plt

def load_metrics(results_dir):
    records = []
    for subdir in os.listdir(results_dir):
        metrics_path = os.path.join(results_dir, subdir, "metrics.csv")
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            records.append(df.iloc[0])
    return pd.DataFrame(records)

def plot_runtime_vs_size(df, out_path):
    df["size"] = df["image"].apply(lambda x: int(x.split("_")[1]))  # e.g., image_128.png
    plt.figure(figsize=(8,6))
    plt.plot(df["size"], df["runtime_sec"], marker="o")
    plt.xlabel("Image Size (N x N)")
    plt.ylabel("Runtime (s)")
    plt.title("BK Runtime vs Image Size")
    plt.savefig(out_path)
    plt.close()

def main():
    results_dir = "bk_batch_results"
    df = load_metrics(results_dir)
    plot_runtime_vs_size(df, "bk_runtime_vs_size.png")
    # ... more plots and markdown report generation

if __name__ == "__main__":
    main()