# Copyright (c) 2024 The Cigma-Sim Authors
# SPDX-License-Identifier: MIT

"""Download experiment datasets (Flickr8k and WMT14 de-en).

Saves HuggingFace datasets to disk so experiment scripts can load them offline via `load_from_disk`.

Usage::

    python scripts/download_datasets.py --flickr8k datasets/Flickr8k
    python scripts/download_datasets.py --wmt14 datasets/wmt14/de-en
    python scripts/download_datasets.py --flickr8k datasets/Flickr8k --wmt14 datasets/wmt14/de-en
"""

import argparse
import sys


def download_flickr8k(output_dir: str) -> None:
    """Download Flickr8k dataset from HuggingFace Hub."""
    from datasets import load_dataset

    remote_path = "jxie/flickr8k"

    print(f"Downloading {remote_path} to {output_dir} ...")
    ds = load_dataset(remote_path)
    ds.save_to_disk(output_dir)
    print(f"Flickr8k saved to {output_dir}")


def download_wmt14(output_dir: str) -> None:
    """Download WMT14 de-en validation set from HuggingFace Hub."""
    from datasets import load_dataset

    remote_path = "wmt/wmt14"
    subset_name = "de-en"

    print(f"Downloading {remote_path} {subset_name} to {output_dir} ...")
    ds = load_dataset(remote_path, subset_name)
    ds.save_to_disk(output_dir)
    print(f"WMT14 saved to {output_dir}")


def main() -> None:
    """Parse arguments and download requested datasets."""
    parser = argparse.ArgumentParser(
        description="Download experiment datasets",
    )
    parser.add_argument(
        "--flickr8k",
        type=str,
        default=None,
        metavar="DIR",
        help="Download Flickr8k dataset to DIR",
    )
    parser.add_argument(
        "--wmt14",
        type=str,
        default=None,
        metavar="DIR",
        help="Download WMT14 de-en validation set to DIR",
    )
    args = parser.parse_args()

    if args.flickr8k is None and args.wmt14 is None:
        parser.print_help()
        sys.exit(1)

    if args.flickr8k is not None:
        download_flickr8k(args.flickr8k)

    if args.wmt14 is not None:
        download_wmt14(args.wmt14)


if __name__ == "__main__":
    main()
