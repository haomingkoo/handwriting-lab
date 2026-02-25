"""Download MNIST and write it in this project's processed PNG+CSV format."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from torchvision import datasets


def _write_split(
    split_name: str,
    dataset,
    output_dir: Path,
    max_samples: int | None = None,
) -> None:
    rows: list[dict[str, int | str]] = []
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    for idx in range(n_samples):
        image, label = dataset[idx]
        rel_path = Path(split_name) / str(label) / f"{idx:05d}.png"
        abs_path = output_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(abs_path)
        rows.append({"filepath": str(rel_path), "label": int(label)})

        if (idx + 1) % 2000 == 0:
            print(f"[{split_name}] wrote {idx + 1}/{n_samples}")

    pd.DataFrame(rows).to_csv(output_dir / f"{split_name}.csv", index=False)
    print(f"[{split_name}] wrote CSV with {len(rows)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/processed/mnist-pngs-data-aisg-processed",
        help="Output directory for train/test PNGs and CSV files.",
    )
    parser.add_argument(
        "--download-dir",
        default="data/raw/torchvision",
        help="Directory where torchvision caches MNIST files.",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional cap for number of training samples.",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Optional cap for number of test samples.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    download_dir = Path(args.download_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    # Prefer stable mirrors to avoid intermittent checksum failures.
    datasets.MNIST.mirrors = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ]

    train_set = datasets.MNIST(root=str(download_dir), train=True, download=True)
    test_set = datasets.MNIST(root=str(download_dir), train=False, download=True)

    _write_split("train", train_set, output_dir, args.max_train)
    _write_split("test", test_set, output_dir, args.max_test)

    print(f"Processed dataset ready at: {output_dir}")


if __name__ == "__main__":
    main()
