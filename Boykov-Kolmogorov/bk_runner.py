import argparse
import os
import time
import numpy as np
from PIL import Image
from working4 import segment_image_with_bk
from simplified import build_grid_graph

# Always use absolute path for the central CSV, and ensure the directory exists
CENTRAL_DIR = os.path.join(os.path.dirname(__file__), "Boykov-Kolmogorov")
CENTRAL_CSV = os.path.join(CENTRAL_DIR, "bk_all_metrics.csv")
os.makedirs(CENTRAL_DIR, exist_ok=True)


def append_to_central_csv(header, row):
    # Always open in append mode, but create the file if it doesn't exist
    file_exists = os.path.isfile(CENTRAL_CSV)
    with open(CENTRAL_CSV, "a", encoding="utf-8") as f:
        if not file_exists or os.path.getsize(CENTRAL_CSV) == 0:
            f.write(header + "\n")
        f.write(row + "\n")


def run_bk_grid_no_folders(size):
    from simplified import BK  # ensure BK is imported
    N = size
    H = W = N
    g = build_grid_graph(H, W)
    V = 2 + H * W
    E = sum(len(g.adj[u]) for u in range(g.n))
    start = time.time()
    flow = g.max_flow(0, 1)
    runtime = time.time() - start
    header = "input_type,input_name,vertices,edges,max_flow,runtime_sec"
    row = f"grid_graph,{N}x{N},{V},{E},{flow},{runtime:.6f}"
    append_to_central_csv(header, row)
    print(f"BK on grid graph {N}x{N}: max_flow={flow}, runtime={runtime:.4f}s, vertices={V}, edges={E}")


def main():
    parser = argparse.ArgumentParser(description="Run BK segmentation on an image or synthetic grid graph.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to input image")
    group.add_argument("--size", type=int, help="Grid size (N for N x N synthetic graph)")
    parser.add_argument("--no_folders", action="store_true", help="Do not create output folders, just update central CSV (for sweeps)")
    args = parser.parse_args()

    if args.no_folders and args.size:
        run_bk_grid_no_folders(args.size)
        return

    if args.image:
        # Image mode
        img = Image.open(args.image).convert("RGB")
        arr = np.array(img)
        start = time.time()
        mask = segment_image_with_bk(arr, auto_seed=True)
        runtime = time.time() - start
        # Save mask if needed (optional, not required for A)
        # mask_path = os.path.join(os.path.dirname(__file__), "Boykov-Kolmogorov", f"mask_{os.path.splitext(os.path.basename(args.image))[0]}.png")
        # Image.fromarray(mask).save(mask_path)
        # Save metrics to central CSV
        header = "input_type,input_name,runtime_sec"
        row = f"image,{os.path.basename(args.image)},{runtime:.6f}"
        append_to_central_csv(header, row)
        print(f"BK image run complete. Runtime: {runtime:.4f} seconds")
    elif args.size:
        if args.no_folders:
            run_bk_grid_no_folders(args.size)
        else:
            # Synthetic grid graph mode (default: with folders for backward compatibility)
            out_dir = os.path.join(os.path.dirname(__file__), "Boykov-Kolmogorov", f"grid_{args.size}")
            os.makedirs(out_dir, exist_ok=True)
            N = args.size
            H = W = N
            g = build_grid_graph(H, W)
            V = 2 + H * W
            E = sum(len(g.adj[u]) for u in range(g.n))
            start = time.time()
            flow = g.max_flow(0, 1)
            runtime = time.time() - start
            metrics_path = os.path.join(out_dir, "metrics.csv")
            with open(metrics_path, "w") as f:
                f.write("input_type,input_name,vertices,edges,max_flow,runtime_sec\n")
                f.write(f"grid_graph,{N}x{N},{V},{E},{flow},{runtime:.6f}\n")
            print(f"BK on grid graph {N}x{N}: max_flow={flow}, runtime={runtime:.4f}s, vertices={V}, edges={E}")
    else:
        raise ValueError("Either --image or --size must be specified.")


if __name__ == "__main__":
    main()