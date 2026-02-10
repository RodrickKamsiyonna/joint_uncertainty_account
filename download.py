"""
Dataset Downloader for SSL Training
Supports: cifar10, imagenet100, imagenette-160

Usage:
    python download.py --dataset cifar10        --root /tmp/data
    python download.py --dataset imagenet100    --root /tmp/data
    python download.py --dataset imagenette-160 --root /tmp/data

    # Use Kaggle source instead (needs API key):
    python download.py --dataset imagenet100 --root /tmp/data \\
        --source kaggle --kaggle_username YOU --kaggle_key YOUR_KEY

Output structure (ImageFolder-compatible):
    <root>/<dataset>/train/<class_name>/<idx>.jpg
    <root>/<dataset>/val/<class_name>/<idx>.jpg
"""

import os
import sys
import shutil
import pickle
import tarfile
import argparse
import subprocess
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd):
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def ensure_pip(*packages):
    for pkg in packages:
        module = pkg.split("[")[0].replace("-", "_")
        try:
            __import__(module)
        except ImportError:
            print(f"  Installing {pkg} ...")
            run(f"{sys.executable} -m pip install -q {pkg}")


def download_url(url, dest_path):
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    run(f'wget -q --show-progress -O "{dest_path}" "{url}"')


def extract_tar(archive, dest):
    os.makedirs(dest, exist_ok=True)
    print(f"  Extracting {archive} → {dest}")
    with tarfile.open(archive) as tf:
        tf.extractall(dest)


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10
# ─────────────────────────────────────────────────────────────────────────────

def download_cifar10(root: str, **kwargs):
    """
    Downloads CIFAR-10 via torchvision and converts to ImageFolder layout.
        <root>/cifar10/train/<classname>/<idx>.png
        <root>/cifar10/val/<classname>/<idx>.png
    """
    ensure_pip("torchvision", "Pillow")
    import torchvision
    from PIL import Image

    dataset_root = Path(root) / "cifar10"
    raw_dir = dataset_root / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]

    print("  Downloading CIFAR-10 via torchvision ...")
    torchvision.datasets.CIFAR10(root=str(raw_dir), train=True,  download=True)
    torchvision.datasets.CIFAR10(root=str(raw_dir), train=False, download=True)

    cifar_dir = raw_dir / "cifar-10-batches-py"

    def load_batch(path):
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        imgs   = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = d[b"labels"]
        return imgs, labels

    def save_split(split_name, batch_files):
        counters = {c: 0 for c in class_names}
        for bf in batch_files:
            imgs, labels = load_batch(cifar_dir / bf)
            for img_arr, label in zip(imgs, labels):
                cls = class_names[label]
                out_dir = dataset_root / split_name / cls
                out_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img_arr).save(out_dir / f"{counters[cls]:05d}.png")
                counters[cls] += 1
        print(f"  Saved {sum(counters.values()):,} images → {dataset_root / split_name}")

    print("  Converting train split ...")
    save_split("train", [f"data_batch_{i}" for i in range(1, 6)])
    print("  Converting val split ...")
    save_split("val", ["test_batch"])

    shutil.rmtree(raw_dir)
    print(f"\n  CIFAR-10 ready at: {dataset_root}")
    print(f"  --train_dir {dataset_root}/train  --val_dir {dataset_root}/val")


# ─────────────────────────────────────────────────────────────────────────────
# ImageNette-160
# ─────────────────────────────────────────────────────────────────────────────

def download_imagenette160(root: str, **kwargs):
    """
    Downloads the 160-px ImageNette tarball from fast.ai (~94 MB).
    Already in ImageFolder layout.
    """
    URL          = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    dataset_root = Path(root) / "imagenette-160"
    archive      = Path(root) / "_tmp_imagenette.tgz"
    tmp_extract  = Path(root) / "_tmp_imagenette_extract"

    dataset_root.mkdir(parents=True, exist_ok=True)

    print("  Downloading imagenette2-160.tgz ...")
    download_url(URL, str(archive))
    extract_tar(str(archive), str(tmp_extract))
    archive.unlink()

    extracted = tmp_extract / "imagenette2-160"
    for split in ("train", "val"):
        src = extracted / split
        dst = dataset_root / split
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        n = sum(1 for p in dst.rglob("*") if p.is_file())
        print(f"  {split}: {n:,} images")

    shutil.rmtree(tmp_extract, ignore_errors=True)
    print(f"\n  ImageNette-160 ready at: {dataset_root}")
    print(f"  --train_dir {dataset_root}/train  --val_dir {dataset_root}/val")


# ─────────────────────────────────────────────────────────────────────────────
# ImageNet-100 — Source 1: HuggingFace  (no login, ~5 GB)
# ─────────────────────────────────────────────────────────────────────────────

def _imagenet100_from_huggingface(root: str):
    """
    Downloads clane9/imagenet-100 from HuggingFace Hub.
    - No account / token required
    - 127k train + 5k val images, pre-resized to 160 px short side
    - Saves as JPEG files in ImageFolder layout
    """
    ensure_pip("datasets", "Pillow")
    from datasets import load_dataset

    dataset_root = Path(root) / "imagenet100"

    # Resume-safe: skip if already complete
    if (dataset_root / "train").exists() and (dataset_root / "val").exists():
        n_train = sum(1 for p in (dataset_root / "train").rglob("*") if p.is_file())
        n_val   = sum(1 for p in (dataset_root / "val").rglob("*")   if p.is_file())
        if n_train > 100_000 and n_val > 4_000:
            print(f"  Already downloaded: {n_train:,} train, {n_val:,} val. Skipping.")
            print(f"  --train_dir {dataset_root}/train  --val_dir {dataset_root}/val")
            return

    print("  Loading clane9/imagenet-100 from HuggingFace ...")
    print("  (No login needed. Expect ~5 GB download, may take 10–30 min.)")

    ds = load_dataset("clane9/imagenet-100", trust_remote_code=True)

    # Build integer-label → safe folder name mapping
    label_names = ds["train"].features["label"].names

    def safe_name(name):
        # Use only the first descriptor before the comma, replace spaces/slashes
        return name.split(",")[0].strip().replace(" ", "_").replace("/", "_")[:40]

    id_to_folder = {i: safe_name(n) for i, n in enumerate(label_names)}

    split_map = {"train": "train", "validation": "val"}

    for hf_split, folder_split in split_map.items():
        if hf_split not in ds:
            print(f"  WARNING: split '{hf_split}' not in dataset, skipping.")
            continue

        split_ds   = ds[hf_split]
        split_path = dataset_root / folder_split
        counters   = {}

        print(f"  Writing {len(split_ds):,} {folder_split} images ...")
        for idx, sample in enumerate(split_ds):
            label     = sample["label"]
            class_dir = split_path / id_to_folder[label]
            class_dir.mkdir(parents=True, exist_ok=True)

            img = sample["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")

            i = counters.get(label, 0)
            img.save(class_dir / f"{i:06d}.jpg", quality=92)
            counters[label] = i + 1

            if (idx + 1) % 10_000 == 0:
                print(f"    {idx + 1:,} / {len(split_ds):,} done ...")

        print(f"  {folder_split}: {sum(counters.values()):,} images saved")

    print(f"\n  ImageNet-100 (HuggingFace) ready at: {dataset_root}")
    print(f"  --train_dir {dataset_root}/train  --val_dir {dataset_root}/val")


# ─────────────────────────────────────────────────────────────────────────────
# ImageNet-100 — Source 2: Kaggle  (needs API key, full-res, ~13 GB)
# ─────────────────────────────────────────────────────────────────────────────

def _imagenet100_from_kaggle(root: str, kaggle_username: str = None, kaggle_key: str = None):
    """
    Downloads ambityga/imagenet100 from Kaggle (~13 GB, full resolution).

    Get your key at: https://www.kaggle.com/settings → API → Create New Token
    Either pass --kaggle_username / --kaggle_key  OR  place kaggle.json at:
        ~/.kaggle/kaggle.json
    """
    ensure_pip("kaggle")

    # Write credentials if supplied inline
    if kaggle_username and kaggle_key:
        import json
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        cred_path  = kaggle_dir / "kaggle.json"
        cred_path.write_text(json.dumps({"username": kaggle_username, "key": kaggle_key}))
        cred_path.chmod(0o600)
        print(f"  Wrote Kaggle credentials to {cred_path}")
    elif not (Path.home() / ".kaggle" / "kaggle.json").exists():
        print("\n  ERROR: No Kaggle credentials found.")
        print("  Either:")
        print("    1. Pass  --kaggle_username YOU  --kaggle_key YOUR_KEY")
        print("    2. Place your kaggle.json at ~/.kaggle/kaggle.json")
        print("  Get your key at: https://www.kaggle.com/settings → API → Create New Token\n")
        sys.exit(1)

    dataset_root = Path(root) / "imagenet100"
    tmp          = Path(root) / "_tmp_imagenet100_kaggle"
    tmp.mkdir(parents=True, exist_ok=True)

    print("  Downloading ambityga/imagenet100 from Kaggle (~13 GB) ...")
    run(f"kaggle datasets download -d ambityga/imagenet100 -p {tmp} --unzip")

    # The Kaggle dataset unpacks as:
    #   train.X1/ train.X2/ train.X3/ train.X4/  ← four shards of train images
    #   val.X/                                    ← validation images
    # Each already has synset sub-dirs (n01440764/ etc.) → valid ImageFolder.
    # We merge the four train shards into one train/ folder.

    print("  Merging train shards ...")
    for shard in sorted(tmp.glob("train.X*")):
        for synset_dir in sorted(shard.iterdir()):
            if not synset_dir.is_dir():
                continue
            dst = dataset_root / "train" / synset_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for img in synset_dir.iterdir():
                shutil.move(str(img), dst / img.name)

    print("  Moving val split ...")
    val_src = tmp / "val.X"
    if val_src.exists():
        for synset_dir in sorted(val_src.iterdir()):
            if not synset_dir.is_dir():
                continue
            dst = dataset_root / "val" / synset_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for img in synset_dir.iterdir():
                shutil.move(str(img), dst / img.name)

    shutil.rmtree(tmp, ignore_errors=True)

    n_train = sum(1 for p in (dataset_root / "train").rglob("*") if p.is_file())
    n_val   = sum(1 for p in (dataset_root / "val").rglob("*")   if p.is_file())
    print(f"  train: {n_train:,} images  |  val: {n_val:,} images")
    print(f"\n  ImageNet-100 (Kaggle) ready at: {dataset_root}")
    print(f"  --train_dir {dataset_root}/train  --val_dir {dataset_root}/val")


def download_imagenet100(root: str, source: str = "huggingface",
                         kaggle_username: str = None, kaggle_key: str = None, **kwargs):
    if source == "kaggle":
        _imagenet100_from_kaggle(root, kaggle_username, kaggle_key)
    else:
        _imagenet100_from_huggingface(root)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "cifar10":        download_cifar10,
    "imagenet100":    download_imagenet100,
    "imagenette-160": download_imagenette160,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download datasets in ImageFolder-compatible format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()),
                        help="Which dataset to download")
    parser.add_argument("--root",    required=True,
                        help="Root directory where the dataset folder will be created")

    # imagenet100-specific
    parser.add_argument("--source", default="huggingface", choices=["huggingface", "kaggle"],
                        help="Source for imagenet100 (default: huggingface — no login needed)")
    parser.add_argument("--kaggle_username", default=None,
                        help="Kaggle username (only needed with --source kaggle)")
    parser.add_argument("--kaggle_key",      default=None,
                        help="Kaggle API key  (only needed with --source kaggle)")
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    os.makedirs(root, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Root    : {root}")
    if args.dataset == "imagenet100":
        print(f"  Source  : {args.source}")
    print(f"{'='*60}\n")

    DATASETS[args.dataset](
        root=root,
        source=args.source,
        kaggle_username=args.kaggle_username,
        kaggle_key=args.kaggle_key,
    )

    print(f"\n✓ Done.\n")


if __name__ == "__main__":
    main()
