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
import gc
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
# ImageNet-100 — Source 1: HuggingFace streaming  (no login, low RAM)
# ─────────────────────────────────────────────────────────────────────────────

def _imagenet100_from_huggingface(root: str):
    """
    Streams clane9/imagenet-100 from HuggingFace one sample at a time.
    - No account / token required
    - Constant ~1-2 GB RAM regardless of dataset size (streaming=True)
    - 127k train + 5k val images, pre-resized to 160 px short side
    """
    ensure_pip("datasets", "Pillow")
    from datasets import load_dataset

    dataset_root = Path(root) / "imagenet100"

    # ── Resume detection ──────────────────────────────────────────────────────
    # Count existing files per split; skip if already looks complete
    def count_files(p):
        return sum(1 for f in p.rglob("*") if f.is_file()) if p.exists() else 0

    n_train_existing = count_files(dataset_root / "train")
    n_val_existing   = count_files(dataset_root / "val")

    splits_to_do = {}
    if n_train_existing > 100_000:
        print(f"  Train already complete ({n_train_existing:,} files). Skipping.")
    else:
        splits_to_do["train"] = ("train", n_train_existing)

    if n_val_existing > 4_000:
        print(f"  Val already complete ({n_val_existing:,} files). Skipping.")
    else:
        splits_to_do["validation"] = ("val", n_val_existing)

    if not splits_to_do:
        print(f"  --train_dir {dataset_root}/train  --val_dir {dataset_root}/val")
        return

    # ── Fetch label names once (tiny metadata call, not the images) ──────────
    print("  Fetching label names from dataset metadata ...")
    # Load only the info (no data) to get feature names
    meta = load_dataset("clane9/imagenet-100", split="train", streaming=True)
    label_names = meta.features["label"].names

    def safe_name(name):
        return name.split(",")[0].strip().replace(" ", "_").replace("/", "_")[:40]

    id_to_folder = {i: safe_name(n) for i, n in enumerate(label_names)}
    del meta
    gc.collect()

    # ── Stream and write each split ───────────────────────────────────────────
    EXPECTED = {"train": 126_689, "validation": 5_000}   # approx totals

    for hf_split, (folder_split, n_existing) in splits_to_do.items():
        expected  = EXPECTED.get(hf_split, "?")
        split_path = dataset_root / folder_split

        print(f"\n  Streaming {hf_split} split ({expected:,} images expected) ...")
        print(f"  RAM stays low — images written to disk one-by-one.")

        # streaming=True: HuggingFace yields one sample at a time, never
        # materialises the full dataset in RAM.
        stream = load_dataset(
            "clane9/imagenet-100",
            split=hf_split,
            streaming=True,
            trust_remote_code=True,
        )

        counters  = {}   # label_int → count of saved files for that class
        skipped   = 0    # files we already have from a previous run
        idx       = 0

        for sample in stream:
            label     = sample["label"]
            class_dir = split_path / id_to_folder[label]
            class_dir.mkdir(parents=True, exist_ok=True)

            i    = counters.get(label, 0)
            dest = class_dir / f"{i:06d}.jpg"

            # Skip files that already exist (resume support)
            if dest.exists():
                counters[label] = i + 1
                skipped += 1
                idx     += 1
                continue

            img = sample["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(dest, quality=92)

            # Free the PIL image immediately — don't let them accumulate
            del img
            sample.clear()

            counters[label] = i + 1
            idx += 1

            if idx % 5_000 == 0:
                saved = idx - skipped
                print(f"    {idx:,} processed  ({saved:,} saved, {skipped:,} skipped) ...")
                gc.collect()   # nudge Python to release any lingering buffers

        total_saved = sum(counters.values()) - skipped
        print(f"  {folder_split}: {sum(counters.values()):,} total  "
              f"({total_saved:,} newly saved, {skipped:,} already existed)")

        # Force GC between splits
        del stream, counters
        gc.collect()

    print(f"\n  ImageNet-100 (HuggingFace) ready at: {dataset_root}")
    print(f"  --train_dir {dataset_root}/train  --val_dir {dataset_root}/val")


# ─────────────────────────────────────────────────────────────────────────────
# ImageNet-100 — Source 2: Kaggle  (needs API key, full-res, ~13 GB)
# ─────────────────────────────────────────────────────────────────────────────

def _imagenet100_from_kaggle(root: str, kaggle_username: str = None, kaggle_key: str = None):
    """
    Downloads ambityga/imagenet100 from Kaggle (~13 GB, full resolution).
    Get your key: https://www.kaggle.com/settings → API → Create New Token
    """
    ensure_pip("kaggle")

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
        print("  Get your key: https://www.kaggle.com/settings → API → Create New Token\n")
        sys.exit(1)

    dataset_root = Path(root) / "imagenet100"
    tmp          = Path(root) / "_tmp_imagenet100_kaggle"
    tmp.mkdir(parents=True, exist_ok=True)

    print("  Downloading ambityga/imagenet100 from Kaggle (~13 GB) ...")
    run(f"kaggle datasets download -d ambityga/imagenet100 -p {tmp} --unzip")

    # Kaggle layout: train.X1/ train.X2/ train.X3/ train.X4/ + val.X/
    # Each already has synset sub-dirs → merge into one train/ folder
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
    print(f"  train: {n_train:,}  |  val: {n_val:,}")
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
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    parser.add_argument("--root",    required=True,
                        help="Root directory where the dataset folder will be created")
    parser.add_argument("--source",  default="huggingface", choices=["huggingface", "kaggle"],
                        help="Source for imagenet100 (default: huggingface — no login, low RAM)")
    parser.add_argument("--kaggle_username", default=None)
    parser.add_argument("--kaggle_key",      default=None)
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
