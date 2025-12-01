import os
import glob
import subprocess

def main():
    images_dir = "images"  # folder with images
    output_dir = "bk_batch_results"
    os.makedirs(output_dir, exist_ok=True)
    image_files = glob.glob(os.path.join(images_dir, "*.png"))  # or jpg

    for img_path in image_files:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(output_dir, img_name)
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run([
            "python", "bk_runner.py",
            "--image", img_path,
            "--output", out_dir
        ])

if __name__ == "__main__":
    main()