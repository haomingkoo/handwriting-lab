"""This script processes raw MNIST images and saves them in the processed data directory."""
from __future__ import annotations

import shutil
import logging
from pathlib import Path

import hydra
import pandas as pd
import torch.utils.data as torch_data
import torchvision

import mnist
from mnist.general_utils import setup_logging

@hydra.main(version_base=None, config_path="../conf", config_name="process_data.yaml")
def main(args):
    """
    Process raw MNIST images and save them into processed directory.
    
    args : omegaconf.DictConfig
        Hydra-composed config from conf/process_data.yaml and any CLI overrides.
        Example keys you likely have in YAML:
        - raw_data_dir
        - processed_data_dir
        - log_dir
        - log_file (optional)

    """
    # Hydra typically changes cwd to something like: outputs/YYYY-MM-DD/HH-MM-SS
    # Relative paths like "./data/raw", they will resolve relative to
    # the Hydra run directory, which might not be what you want.
    #
    # hydra.utils.to_absolute_path(...) converts a path relative to the ORIGINAL
    # working directory (where you launched the script) into an absolute path.

    raw_data_dir = Path(hydra.utils.to_absolute_path(args.raw_data_dir))
    processed_data_dir = Path(hydra.utils.to_absolute_path(args.processed_data_dir))
    log_dir = Path(hydra.utils.to_absolute_path(args.log_dir))
    
    #  If missing, default to "process_data.log".
    log_file = getattr(args, "log_file", "process_data.log")

    # parents=True: create intermediate directories if needed
    # exist_ok=True: do not error if folder already exists
    log_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging (uses your provided utility + conf/logging.yaml)
    # setup_logging(...):
    # - Loads YAML logging config (handlers/formatters/levels)
    # - Sets file handlers under log_dir (e.g., debug.log/info.log/warn.log)
    #
    # Important: logging_config_path is resolved to an absolute path too.
    # We pass the path to the YAML file and a directory where logs should go.

    setup_logging(
        logging_config_path=hydra.utils.to_absolute_path("conf/logging.yaml"),
        log_dir=hydra.utils.to_absolute_path(args.log_dir),
    )
    logger = logging.getLogger(__name__)
    
    # Structured logs for debugging runs, helpful for Hydra multirun.
    logger.info("Starting MNIST raw image processing.")
    logger.info("raw_data_dir=%s", raw_data_dir)
    logger.info("processed_data_dir=%s", processed_data_dir)
    logger.info("log_dir=%s, log_file=%s", log_dir, log_file)

    # Basic validation
    if not raw_data_dir.exists():
        # Fail early if input folder is wrong.
        logger.error("Raw data directory does not exist: %s", raw_data_dir)
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")
    
    # Find subdirectories under raw_data_dir.
    # Each subdir is assumed to contain train.csv and (optionally) test.csv.
    raw_subdirs = [p for p in raw_data_dir.iterdir() if p.is_dir()]
    if not raw_subdirs:
        logger.warning("No subdirectories found under raw_data_dir: %s", raw_data_dir)

    # Collect all train annotation DataFrames here and concat later.
    combined_anno_train_df_list: list[pd.DataFrame] = []

    # Loop over each raw subdir and process it
    for subdir in raw_subdirs:
        logger.info("Processing directory: %s", subdir)

    # Read train.csv
        train_csv = subdir / "train.csv"
        if not train_csv.exists():
            # If train.csv is missing, we cannot build training data from that subdir.
            logger.warning("Missing train.csv in %s. Skipping.", subdir)
            continue

        # Read the annotation table (probably columns like: path/label/etc).
        # Exception handling here ensures you get stack trace in logs.
        try:
            curr_anno_train_df = pd.read_csv(train_csv)
        except Exception as e:
            logger.exception("Failed to read %s: %s", train_csv, e)
            raise

        combined_anno_train_df_list.append(curr_anno_train_df)

        # Save training images as PNGs
        # MNISTDataset is the custom dataset wrapper.
        # It likely reads train.csv, loads corresponding image files,
        # and returns something like:
        #   (file_name, image_tensor, label)
        #
        # to_tensor=False here suggests MNISTDataset returns tensors already
        # after applying `transform`, or returns something torchvision can handle.
        # (Exact behavior depends on your MNISTDataset implementation.)

        curr_train_dataset = mnist.data_prep.datasets.MNISTDataset(
            str(subdir),
            "train.csv",
            to_tensor=False,
            transform=mnist.data_prep.transforms.MNIST_TRANSFORM_STEPS["train"],
        )

        # DataLoader iterates the dataset. Default batch_size=1 if not specified.
        curr_train_loader = torch_data.DataLoader(curr_train_dataset)

        train_count = 0
        for batch in curr_train_loader:
            # The dataset returns 3 items. The DataLoader batches them.
            # Because batch_size defaults to 1:
            # - train_image_file_name is a list/tuple length 1
            # - train_image is a tensor with leading batch dimension
            # - label is present but ignored here

            train_image_file_name, train_image, _ = batch

            # Construct output path, preserving the relative file structure.
            # Example: processed_data_dir / "train/0/20713.png"
            dest_path = processed_data_dir / train_image_file_name[0]

            # Ensure parent directories exist (e.g., processed/train/0/)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # save_image expects a CHW tensor. train_image[0] removes batch dim.
            torchvision.utils.save_image(train_image[0], str(dest_path))
            train_count += 1

        logger.info("Saved %d train images for %s", train_count, subdir.name)

        # Save test images as PNGs (if test.csv exists)
        test_csv = subdir / "test.csv"
        if test_csv.exists():
            test_dataset = mnist.data_prep.datasets.MNISTDataset(
                str(subdir),
                "test.csv",
                to_tensor=False,
                transform=mnist.data_prep.transforms.MNIST_TRANSFORM_STEPS["test"],
            )
            test_loader = torch_data.DataLoader(test_dataset)

            test_count = 0
            for batch in test_loader:
                test_image_file_name, test_image, _ = batch
                dest_path = processed_data_dir / test_image_file_name[0]
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(test_image[0], str(dest_path))
                test_count += 1

            # Copy test.csv once per subdir (keeps last copy)
            # IMPORTANT: If you have multiple subdirs, this will overwrite repeatedly.
            # In production either:
            # - merge test.csv like train.csv, or
            # - ensure only one raw subdir exists.

            shutil.copy(str(test_csv), str(processed_data_dir / "test.csv"))
            logger.info("Saved %d test images for %s", test_count, subdir.name)
        else:
            logger.warning("No test.csv found in %s. Skipping test split for this subdir.", subdir)
 
    # Merge all train.csv files into a single combined train.csv
    if not combined_anno_train_df_list:
        logger.error("No train.csv files were processed. Nothing to write.")
        raise RuntimeError("No training annotations found. Check raw_data_dir contents.")

    combined_anno_train_df = pd.concat(combined_anno_train_df_list, ignore_index=True)
    out_train_csv = processed_data_dir / "train.csv"

    # index=False prevents pandas from writing an extra index column
    combined_anno_train_df.to_csv(out_train_csv, index=False)
    logger.info("Wrote combined train.csv: %s (rows=%d)", out_train_csv, len(combined_anno_train_df))

    logger.info("All raw data has been processed successfully.")

if __name__ == "__main__":
    # When running as a script, Hydra calls main() and injects args automatically.
    main()
