"""
Dataset Downloader for SSL Training
Supports: cifar10, imagenet100, imagenette-160

Usage:
    python download.py --dataset cifar10        --root /tmp/data
    python download.py --dataset imagenet100    --root /tmp/data
    python download.py --dataset imagenette-160 --root /tmp/data

Output structure (ImageFolder-compatible):
    <root>/<dataset>/train/<class>/<image>.jpg
    <root>/<dataset>/val/<class>/<image>.jpg
"""

import os
import argparse
import shutil
import tarfile
import zipfile
import subprocess
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd, **kwargs):
    print(f"  $ {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kwargs)


def ensure_pip(*packages):
    """Install packages quietly if missing."""
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"  Installing {pkg} ...")
            run(f"{sys.executable} -m pip install -q {pkg}")


def download_url(url, dest_path):
    """Download a URL to dest_path using wget (always available on Kaggle/Colab)."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    run(f'wget -q --show-progress -O "{dest_path}" "{url}"')


def extract_tar(archive, dest):
    os.makedirs(dest, exist_ok=True)
    print(f"  Extracting {archive} → {dest}")
    with tarfile.open(archive) as tf:
        tf.extractall(dest)


def extract_zip(archive, dest):
    os.makedirs(dest, exist_ok=True)
    print(f"  Extracting {archive} → {dest}")
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest)


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10  →  ImageFolder layout
# ─────────────────────────────────────────────────────────────────────────────

def download_cifar10(root: str):
    """
    Downloads CIFAR-10 via torchvision and converts the raw batches into a
    standard ImageFolder tree:
        <root>/cifar10/train/<classname>/<idx>.png
        <root>/cifar10/val/<classname>/<idx>.png
    """
    ensure_pip("torchvision", "Pillow")
    import torchvision
    from PIL import Image
    import pickle
    import numpy as np

    dataset_root = Path(root) / "cifar10"
    raw_dir = dataset_root / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    # torchvision will download + verify the raw batches
    print("  Downloading CIFAR-10 raw batches via torchvision …")
    torchvision.datasets.CIFAR10(root=str(raw_dir), train=True,  download=True)
    torchvision.datasets.CIFAR10(root=str(raw_dir), train=False, download=True)

    cifar_dir = raw_dir / "cifar-10-batches-py"

    def load_batch(path):
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        imgs   = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC
        labels = d[b"labels"]
        return imgs, labels

    def save_split(split_name, batches):
        counters = {c: 0 for c in class_names}
        for batch_file in batches:
            imgs, labels = load_batch(cifar_dir / batch_file)
            for img_arr, label in zip(imgs, labels):
                cls = class_names[label]
                out_dir = dataset_root / split_name / cls
                out_dir.mkdir(parents=True, exist_ok=True)
                idx = counters[cls]
                Image.fromarray(img_arr).save(out_dir / f"{idx:05d}.png")
                counters[cls] += 1
        total = sum(counters.values())
        print(f"  Saved {total:,} images to {dataset_root / split_name}")

    train_batches = [f"data_batch_{i}" for i in range(1, 6)]
    test_batches  = ["test_batch"]

    print("  Converting train split …")
    save_split("train", train_batches)
    print("  Converting val split …")
    save_split("val",   test_batches)

    # Clean up raw download
    shutil.rmtree(raw_dir)
    print(f"  CIFAR-10 ready at {dataset_root}")
    print(f"  Pass --train_dir {dataset_root}/train  --val_dir {dataset_root}/val")


# ─────────────────────────────────────────────────────────────────────────────
# ImageNette-160
# ─────────────────────────────────────────────────────────────────────────────

def download_imagenette160(root: str):
    """
    Downloads the 160-px ImageNette tarball from fast.ai and extracts it.
    The archive already has the correct ImageFolder layout:
        imagenette2-160/train/<class>/...
        imagenette2-160/val/<class>/...
    We just move it into <root>/imagenette-160/{train,val}.
    """
    URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    dataset_root = Path(root) / "imagenette-160"
    archive      = Path(root) / "_tmp_imagenette.tgz"
    tmp_extract  = Path(root) / "_tmp_imagenette_extract"

    dataset_root.mkdir(parents=True, exist_ok=True)

    print("  Downloading imagenette2-160.tgz …")
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
        n = sum(1 for _ in dst.rglob("*") if _.is_file())
        print(f"  {split}: {n:,} images")

    shutil.rmtree(tmp_extract, ignore_errors=True)
    print(f"  ImageNette-160 ready at {dataset_root}")
    print(f"  Pass --train_dir {dataset_root}/train  --val_dir {dataset_root}/val")


# ─────────────────────────────────────────────────────────────────────────────
# ImageNet-100
# ─────────────────────────────────────────────────────────────────────────────

# 100 synset IDs used by the popular ImageNet-100 benchmark split
# (same set used in MoCo v2, DINO, etc. papers)
IMAGENET100_SYNSETS = [
    "n01440764","n01443537","n01484850","n01491361","n01494475",
    "n01498041","n01514668","n01514859","n01518878","n01530575",
    "n01531178","n01532829","n01534433","n01537544","n01558993",
    "n01560419","n01580077","n01582220","n01592084","n01601694",
    "n01608432","n01614925","n01616318","n01622779","n01629819",
    "n01630670","n01631663","n01632458","n01632777","n01641577",
    "n01644373","n01644900","n01664065","n01665541","n01667114",
    "n01667778","n01669191","n01675722","n01677366","n01682714",
    "n01685808","n01687978","n01688243","n01689811","n01692333",
    "n01693334","n01694178","n01695060","n01697457","n01698640",
    "n01704323","n01728572","n01728920","n01729322","n01729977",
    "n01734418","n01735189","n01737021","n01739381","n01740131",
    "n01742172","n01744401","n01748264","n01749939","n01751748",
    "n01753488","n01755581","n01756291","n02012849","n02013706",
    "n02017213","n02025239","n02027492","n02028035","n02033041",
    "n02037110","n02051845","n02056570","n02058221","n02077923",
    "n02085620","n02085782","n02085936","n02086079","n02086240",
    "n02086646","n02086910","n02087046","n02087394","n02088094",
    "n02088238","n02088364","n02088466","n02088632","n02089078",
    "n02089867","n02089973","n02090379","n02090622","n02090721",
]


def download_imagenet100_kaggle(root: str):
    """
    Downloads ImageNet-100 from Kaggle (imagenet-object-localization-challenge)
    by subsetting to the 100 synsets above.

    Prerequisites on Kaggle:
        - The kernel must have the competition dataset attached, OR
        - kaggle.json must be configured for API access.

    On Kaggle notebooks the full ImageNet is already at:
        /kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/
    So we just symlink / copy the 100 classes without re-downloading.
    """
    KAGGLE_ILSVRC = Path("/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC")

    if KAGGLE_ILSVRC.exists():
        print("  Detected Kaggle ILSVRC dataset. Subsetting to 100 classes …")
        _subset_imagenet(KAGGLE_ILSVRC, root)
        return

    # ── Fallback: try Kaggle API ──────────────────────────────────────────────
    print("  Kaggle ILSVRC path not found. Trying Kaggle API download …")
    ensure_pip("kaggle")
    tmp = Path(root) / "_tmp_imagenet100"
    tmp.mkdir(parents=True, exist_ok=True)

    run(
        f"kaggle competitions download "
        f"-c imagenet-object-localization-challenge "
        f"-p {tmp}"
    )
    # The download is a huge zip; extract only what we need
    zip_path = tmp / "imagenet-object-localization-challenge.zip"
    if zip_path.exists():
        print("  Extracting selected synsets from archive (this may take a while) …")
        _extract_synsets_from_zip(zip_path, root)
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        raise FileNotFoundError(
            "Could not find the Kaggle zip. "
            "Please attach the 'imagenet-object-localization-challenge' dataset "
            "to your Kaggle notebook or configure your kaggle.json API key."
        )


def _subset_imagenet(ilsvrc_root: Path, root: str):
    """Copy / hard-link 100 synset folders from a full ILSVRC tree."""
    dataset_root = Path(root) / "imagenet100"
    synset_set = set(IMAGENET100_SYNSETS)

    for split in ("train", "val"):
        src_split = ilsvrc_root / split
        dst_split = dataset_root / split
        if not src_split.exists():
            print(f"  WARNING: {src_split} not found, skipping.")
            continue

        count = 0
        for synset_dir in sorted(src_split.iterdir()):
            if synset_dir.name not in synset_set:
                continue
            dst_dir = dst_split / synset_dir.name
            if dst_dir.exists():
                continue
            # Use symlinks to avoid copying ~130 GB
            dst_dir.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(synset_dir.resolve(), dst_dir)
            count += 1

        n = sum(1 for _ in dst_split.rglob("*") if _.is_file())
        print(f"  {split}: linked {count} synsets  ({n:,} images)")

    print(f"  ImageNet-100 ready at {dataset_root}")
    print(f"  Pass --train_dir {dataset_root}/train  --val_dir {dataset_root}/val")


def _extract_synsets_from_zip(zip_path: Path, root: str):
    """Selectively extract only the 100 synset folders from the competition zip."""
    dataset_root = Path(root) / "imagenet100"
    synset_set = set(IMAGENET100_SYNSETS)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        selected = [
            m for m in members
            if any(f"/{s}/" in m or m.endswith(f"/{s}") for s in synset_set)
        ]
        print(f"  Extracting {len(selected):,} entries …")
        for member in selected:
            zf.extract(member, str(dataset_root))

    print(f"  ImageNet-100 ready at {dataset_root}")


def download_imagenet100(root: str):
    """Entry-point: detects environment and downloads ImageNet-100."""
    print("  Checking for ImageNet-100 sources …")

    # ── Option 1: already on Kaggle ──────────────────────────────────────────
    kaggle_path = Path("/kaggle/input/imagenet-object-localization-challenge")
    if kaggle_path.exists():
        download_imagenet100_kaggle(root)
        return

    # ── Option 2: HuggingFace Hub (imagenet-1k subset, public) ───────────────
    # There is no clean free 100-class subset on HF, so we guide the user.
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║              ImageNet-100 – Manual Step Required             ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print("  ║  ImageNet requires a free account registration:              ║")
    print("  ║  https://www.image-net.org/download.php                      ║")
    print("  ║                                                               ║")
    print("  ║  On Kaggle, attach this dataset to your notebook:            ║")
    print("  ║  imagenet-object-localization-challenge                       ║")
    print("  ║  and re-run:                                                  ║")
    print("  ║    python download.py --dataset imagenet100 --root /tmp/data  ║")
    print("  ║                                                               ║")
    print("  ║  The script will then auto-subset to the 100 classes.         ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()
    sys.exit(0)


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
        description="Download datasets for SSL training (ImageFolder-compatible output).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=list(DATASETS.keys()),
        help="Which dataset to download.",
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory where datasets will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    os.makedirs(root, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Root    : {root}")
    print(f"{'='*60}\n")

    DATASETS[args.dataset](root)

    print(f"\n✓ Done. Dataset saved under {root}/{args.dataset}/\n")


if __name__ == "__main__":
    main()
