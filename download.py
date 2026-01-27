import os
import shutil
import tarfile
import zipfile
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
import torchvision
from torchvision.datasets import CIFAR10
from PIL import Image

def download_cifar10(root):
    """
    Downloads CIFAR10 via torchvision and saves it as a folder structure:
    root/cifar10/train/class_name/image.png
    """
    dataset_dir = Path(root) / "cifar10"
    if dataset_dir.exists():
        print(f"CIFAR-10 directory already exists at {dataset_dir}. Skipping.")
        return

    print("--- Downloading and Extracting CIFAR-10 ---")
    
    # Download raw data using torchvision
    train_set = CIFAR10(root=root, train=True, download=True)
    val_set = CIFAR10(root=root, train=False, download=True)
    
    classes = train_set.classes
    
    def save_images(dataset, split_name):
        split_dir = dataset_dir / split_name
        os.makedirs(split_dir, exist_ok=True)
        
        # Create class subdirectories
        for cls in classes:
            os.makedirs(split_dir / cls, exist_ok=True)
            
        print(f"Saving {split_name} images to disk...")
        for i in tqdm(range(len(dataset))):
            img, label_idx = dataset[i]
            label_name = classes[label_idx]
            # Save as PNG
            img.save(split_dir / label_name / f"{i}.png")

    save_images(train_set, "train")
    save_images(val_set, "val")
    
    shutil.rmtree(Path(root) / "cifar-10-batches-py", ignore_errors=True)
    os.remove(Path(root) / "cifar-10-python.tar.gz")
    print(f"CIFAR-10 ready at: {dataset_dir}")

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
    
    # Download
    subprocess.run(["wget", url, "-O", str(tar_path)], check=True)
    
    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=root)
    
    # Rename folder to standard name
    extracted_folder = Path(root) / "imagenette2-160"
    extracted_folder.rename(dataset_dir)
    
    os.remove(tar_path)
    print(f"Imagenette ready at: {dataset_dir}")

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
    
    # Using the popular 'ambityga/imagenet100' dataset
    kaggle_dataset = "ambityga/imagenet100"
    
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", root], check=True)
    except FileNotFoundError:
        print("Error: 'kaggle' command not found. Please install with `pip install kaggle`.")
        return
    except subprocess.CalledProcessError:
        print("Error: Download failed. Check your API key.")
        return

    zip_path = Path(root) / "imagenet100.zip"
    print("Extracting ImageNet-100...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_dir)
    
    os.remove(zip_path)
    print(f"ImageNet-100 ready at: {dataset_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "imagenette", "imagenet100", "all"])
    parser.add_argument("--root", type=str, default="./data", help="Root folder to store datasets")
    args = parser.parse_args()

    if args.dataset in ["cifar10", "all"]:
        download_cifar10(args.root)
    if args.dataset in ["imagenette", "all"]:
        download_imagenette(args.root)
    if args.dataset in ["imagenet100", "all"]:
        download_imagenet100(args.root)
