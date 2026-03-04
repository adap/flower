#!/usr/bin/env python3
"""
Download CIFAR-10 and save as pickles expected by FRIDA's loader.

Creates:
  - <out-dir>/train_data.pkl  -> (train_images, train_labels)
  - <out-dir>/test_data.pkl   -> (test_images, test_labels)
Optionally:
  - <out-dir>/classes.pkl     -> list[str]

Where:
  train_images/test_images: np.ndarray uint8, shape (N, 32, 32, 3)
  train_labels/test_labels: np.ndarray int64, shape (N,)
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np

try:
    from torchvision.datasets import CIFAR10
except ImportError as e:
    raise SystemExit("torchvision is required. Install with: pip install torchvision") from e


def dataset_to_arrays(ds: CIFAR10) -> tuple[np.ndarray, np.ndarray]:
    images = np.asarray(ds.data, dtype=np.uint8)          # (N, 32, 32, 3)
    labels = np.asarray(ds.targets, dtype=np.int64)       # (N,)
    return images, labels


def atomic_pickle_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/cifar10"),
                   help='Output dir (default: "data/cifar10")')
    p.add_argument("--torchvision-root", type=Path, default=None,
                   help='Where torchvision stores downloads (default: <out-dir>/torchvision)')
    p.add_argument("--no-classes", action="store_true",
                   help="Do not write classes.pkl")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing .pkl files if they exist")
    args = p.parse_args()

    out_dir: Path = args.out_dir
    tv_root: Path = args.torchvision_root or (out_dir / "torchvision")

    train_path = out_dir / "train_data.pkl"
    test_path = out_dir / "test_data.pkl"
    classes_path = out_dir / "classes.pkl"

    if not args.force:
        existing = [p for p in (train_path, test_path) if p.exists()]
        if existing:
            raise SystemExit(
                "Refusing to overwrite existing files:\n"
                + "\n".join(f"  - {x}" for x in existing)
                + "\nRe-run with --force to overwrite."
            )

    print(f"Downloading CIFAR-10 via torchvision into: {tv_root}")
    train_ds = CIFAR10(root=str(tv_root), train=True, download=True)
    test_ds = CIFAR10(root=str(tv_root), train=False, download=True)

    train_images, train_labels = dataset_to_arrays(train_ds)
    test_images, test_labels = dataset_to_arrays(test_ds)

    print(f"Saving: {train_path}  ({train_images.shape}, {train_labels.shape})")
    atomic_pickle_dump(train_path, (train_images, train_labels))

    print(f"Saving: {test_path}   ({test_images.shape}, {test_labels.shape})")
    atomic_pickle_dump(test_path, (test_images, test_labels))

    if not args.no_classes:
        print(f"Saving: {classes_path}  ({len(train_ds.classes)} classes)")
        atomic_pickle_dump(classes_path, list(train_ds.classes))

    print("Done.")


if __name__ == "__main__":
    main()
