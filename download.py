import os
import shutil
import tarfile
import zipfile
import subprocess
import argparse
import gc  # Essential for freeing memory
from pathlib import Path
from tqdm import tqdm
from torchvision.datasets import CIFAR10

def download_cifar10(root):
    """
    Downloads CIFAR10 via torchvision and saves it as a folder structure.
    OPTIMIZED: Loads/clears train and val sets sequentially to save RAM.
    """
    dataset_dir = Path(root) / "cifar10"
    if dataset_dir.exists():
        print(f"CIFAR-10 directory already exists at {dataset_dir}. Skipping.")
        return

    print("--- Downloading and Extracting CIFAR-10 ---")
    
    # Helper function to handle one split at a time
    def process_split(train_bool, split_name):
        print(f"Loading CIFAR-10 {split_name} set into memory...")
        # Download/Load ONLY this split
        # We assume download=True so torchvision checks integrity, 
        # but it won't re-download if the file exists.
        dataset = CIFAR10(root=root, train=train_bool, download=True)
        
        classes = dataset.classes
        split_dir = dataset_dir / split_name
        os.makedirs(split_dir, exist_ok=True)
        
        # Create class subdirectories
        for cls in classes:
            os.makedirs(split_dir / cls, exist_ok=True)
            
        print(f"Saving {split_name} images to disk...")
        for i in tqdm(range(len(dataset))):
            img, label_idx = dataset[i]
            label_name = classes[label_idx]
            img.save(split_dir / label_name / f"{i}.png")
        
        # CRITICAL: Delete the dataset object and free memory
        del dataset
        gc.collect()

    # 1. Process Train
    process_split(True, "train")
    
    # 2. Process Val
    process_split(False, "val")

    # Cleanup raw files
    print("Cleaning up CIFAR-10 raw archives...")
    shutil.rmtree(Path(root) / "cifar-10-batches-py", ignore_errors=True)
    try:
        os.remove(Path(root) / "cifar-10-python.tar.gz")
    except FileNotFoundError:
        pass
        
    print(f"CIFAR-10 ready at: {dataset_dir}")
    print("-" * 30)

def download_imagenette(root):
    """
    Downloads Imagenette (160px version) from fast.ai.
    """
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    dataset_dir = Path(root) / "imagenette"
    tar_path = Path(root) / "imagenette2-160.tgz"
    
    if dataset_dir.exists():
        print(f"Imagenette directory already exists at {dataset_dir}. Skipping.")
        return

    print("--- Downloading Imagenette ---")
    os.makedirs(root, exist_ok=True)
    
    # Check if wget is available, otherwise warn
    if shutil.which("wget") is None:
        print("Error: 'wget' is not installed or not in PATH.")
        return

    # Download
    subprocess.run(["wget", url, "-O", str(tar_path)], check=True)
    
    print("Extracting Imagenette...")
    # 'r:gz' opens the tarfile with gzip compression
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=root)
    
    # Rename folder to standard name
    extracted_folder = Path(root) / "imagenette2-160"
    if extracted_folder.exists():
        extracted_folder.rename(dataset_dir)
    
    os.remove(tar_path)
    gc.collect() # Force cleanup
    print(f"Imagenette ready at: {dataset_dir}")
    print("-" * 30)

def download_imagenet100(root):
    """
    Downloads ImageNet-100 from Kaggle.
    Requires 'kaggle' pip package and ~/.kaggle/kaggle.json
    """
    dataset_dir = Path(root) / "imagenet100"
    if dataset_dir.exists():
        print(f"ImageNet-100 directory already exists at {dataset_dir}. Skipping.")
        return

    print("--- Downloading ImageNet-100 (via Kaggle API) ---")
    print("Ensure you have your kaggle.json set up!")
    
    os.makedirs(root, exist_ok=True)
    
    kaggle_dataset = "ambityga/imagenet100"
    
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", root], check=True)
    except FileNotFoundError:
        print("Error: 'kaggle' command not found. Please install with `pip install kaggle`.")
        return
    except subprocess.CalledProcessError:
        print("Error: Download failed. Check your API key and internet connection.")
        return

    # Kaggle downloads it as the dataset name usually, but here we expect a zip
    # Note: Kaggle sometimes downloads as 'imagenet100.zip' depending on the dataset slug
    zip_path = Path(root) / "imagenet100.zip"
    
    if not zip_path.exists():
        print(f"Expected zip file at {zip_path} not found. Check the download filename.")
        return

    print("Extracting ImageNet-100...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    
    os.remove(zip_path)
    gc.collect() # Force cleanup
    print(f"ImageNet-100 ready at: {dataset_dir}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "imagenette", "imagenet100", "all"])
    parser.add_argument("--root", type=str, default="./data", help="Root folder to store datasets")
    args = parser.parse_args()

    # Ensure root exists
    os.makedirs(args.root, exist_ok=True)

    if args.dataset in ["cifar10", "all"]:
        download_cifar10(args.root)
        gc.collect() # Cleanup after function returns
        
    if args.dataset in ["imagenette", "all"]:
        download_imagenette(args.root)
        gc.collect()
        
    if args.dataset in ["imagenet100", "all"]:
        download_imagenet100(args.root)
        gc.collect()
