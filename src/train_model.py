"""
This script is for training a model on the MNIST dataset.

Key ideas:
- Hydra loads conf/train_model.yaml into `args` (an OmegaConf DictConfig).
- MLflow is used to track runs, params, metrics, and artifacts.
- Script is Cloud Run friendly with relative project paths.
- Logging config is resolved via hydra.utils.get_original_cwd() (repo root), not Hydra run dir.
- Produces a plain PyTorch state_dict artifact at ./data/model.pth (for your pipeline).
- Logs that ./data/model.pth into MLflow under artifact path "data/" (so pull-model can download it).
- Optionally logs an MLflow-native PyTorch model ("model/") and registers it (for registry workflows).

"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
import mlflow
import mnist
import torch


def _to_bool(value) -> bool:
    """Convert config values to bool while handling string overrides safely."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# pylint: disable = no-value-for-parameter
@hydra.main(version_base=None, config_path="../conf", config_name="train_model.yaml")
def main(args):
    """
    Hydra:
    - Reads ../conf/train_model.yaml
    - Passes it in as `args` (OmegaConf DictConfig)
    - Changes the working directory to a Hydra-managed run folder
      (important for file paths; use hydra.utils.get_original_cwd() if needed)
    """

    # Configure logging using project utility + conf/logging.yaml.
    # hydra.utils.get_original_cwd() is used because Hydra changes cwd at runtime.

    repo_root = Path(hydra.utils.get_original_cwd())

    mnist.general_utils.setup_logging(
        logging_config_path=str(repo_root / "conf" / "logging.yaml"),
        log_dir=args.get("log_dir", None),
    )

    logger = logging.getLogger(__name__)
    logger.info("Logging configured.")

    # MLflow settings come from YAML (with defaults if missing).
    # If  run locally without an MLflow server, tracking_uri points to ./mlruns.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or args.get("mlflow_tracking_uri", "http://127.0.0.1:5000")
    
    exp_name = args.get("mlflow_exp_name", "mnist")
    run_name = args.get("mlflow_run_name", "train_model")

    logger.info("MLflow tracking_uri resolved to: %s", tracking_uri)
    logger.info("MLflow exp_name=%s run_name=%s", exp_name, run_name)


    # Initialise MLflow run.
    # - resume controls whether you continue an existing run/checkpoint conceptually.
    # - step_offset supports continuing epoch numbering when resuming.
    mlflow_init_status, mlflow_run, step_offset = mnist.general_utils.mlflow_init(
        tracking_uri=tracking_uri,
        exp_name=exp_name,
        run_name=run_name,
        setup_mlflow=args.get("setup_mlflow", True),
        autolog=args.get("mlflow_autolog", False),
        resume=args.get("resume", False),
    )

    # Always log the real MLflow identifiers as early as possible.
    if mlflow_init_status and mlflow_run is not None:
        logger.info("REAL_MLFLOW_RUN_ID=%s", mlflow_run.info.run_id)
        logger.info("MLFLOW_EXPERIMENT_ID=%s", mlflow_run.info.experiment_id)
        logger.info("MLFLOW_EXPERIMENT_NAME=%s", exp_name)
        logger.info("MLFLOW_TRACKING_URI=%s", mlflow.get_tracking_uri())
    else:
        logger.warning("MLflow not initialised. Runs/artifacts will not be tracked.")

    # If MLflow is disabled or failed, ensure step_offset is 0 so epoch loop is normal.
    if not mlflow_init_status:
        step_offset = 0

    # Reproducibility for torch random operations (init weights, shuffling, etc.).
    torch.manual_seed(int(args["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args["seed"]))


    # Device selection:
    # - cuda if allowed and available
    # - else mps (Apple) if allowed and available
    # - else cpu
    use_cuda = not args["no_cuda"] and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif not args["no_mps"] and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Read optimizer-related hyperparameters from config (or defaults).
    opt_name = str(args.get("optimizer", "adam")).lower()
    weight_decay = float(args.get("weight_decay", 0.0))
    momentum = float(args.get("momentum", 0.0))
    enable_train_augmentation = _to_bool(args.get("enable_train_augmentation", False))
    train_rotation_degrees = float(args.get("train_rotation_degrees", 0.0))
    train_rotation_prob = float(args.get("train_rotation_prob", 0.0))
    train_affine_prob = float(args.get("train_affine_prob", 0.0))
    train_affine_translate_x = float(args.get("train_affine_translate_x", 0.0))
    train_affine_translate_y = float(args.get("train_affine_translate_y", 0.0))
    train_affine_scale_min = float(args.get("train_affine_scale_min", 1.0))
    train_affine_scale_max = float(args.get("train_affine_scale_max", 1.0))
    train_affine_shear_degrees = float(args.get("train_affine_shear_degrees", 0.0))
    train_perspective_prob = float(args.get("train_perspective_prob", 0.0))
    train_perspective_distortion_scale = float(
        args.get("train_perspective_distortion_scale", 0.0)
    )
    train_invert_prob = float(args.get("train_invert_prob", 0.0))

    # Log the trial hyperparameters into the console logs.
    # This is useful to confirm Hydra multirun is actually varying these values.
    logger.info(
        (
            "Trial hyperparams: optimizer=%s lr=%s gamma=%s weight_decay=%s "
            "momentum=%s augment=%s rotation_degrees=%s rotation_prob=%s "
            "affine_prob=%s affine_translate_x=%s affine_translate_y=%s "
            "affine_scale_min=%s affine_scale_max=%s affine_shear_degrees=%s "
            "perspective_prob=%s perspective_distortion_scale=%s invert_prob=%s"
        ),
        opt_name,
        float(args["lr"]),
        float(args["gamma"]),
        weight_decay,
        momentum,
        enable_train_augmentation,
        train_rotation_degrees,
        train_rotation_prob,
        train_affine_prob,
        train_affine_translate_x,
        train_affine_translate_y,
        train_affine_scale_min,
        train_affine_scale_max,
        train_affine_shear_degrees,
        train_perspective_prob,
        train_perspective_distortion_scale,
        train_invert_prob,
    )

    # Log hyperparameters to MLflow (so each run records what it used).
    mnist.general_utils.mlflow_log(
        mlflow_init_status,
        "log_params",
        params={
            "seed": args["seed"],
            "epochs": args["epochs"],
            "lr": args["lr"],
            "gamma": args["gamma"],
            "train_bs": args["train_bs"],
            "test_bs": args["test_bs"],
            "model_checkpoint_interval": args["model_checkpoint_interval"],
            "device": str(device),
            "optimizer": opt_name,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "enable_train_augmentation": enable_train_augmentation,
            "train_rotation_degrees": train_rotation_degrees,
            "train_rotation_prob": train_rotation_prob,
            "train_affine_prob": train_affine_prob,
            "train_affine_translate_x": train_affine_translate_x,
            "train_affine_translate_y": train_affine_translate_y,
            "train_affine_scale_min": train_affine_scale_min,
            "train_affine_scale_max": train_affine_scale_max,
            "train_affine_shear_degrees": train_affine_shear_degrees,
            "train_perspective_prob": train_perspective_prob,
            "train_perspective_distortion_scale": train_perspective_distortion_scale,
            "train_invert_prob": train_invert_prob,
        },
    )

    # DataLoader settings.
    # - shuffle=True for train so SGD sees different ordering each epoch
    # - test does not need shuffle
    train_kwargs = {"batch_size": args["train_bs"], "shuffle": True}
    test_kwargs = {"batch_size": args["test_bs"]}
    
    # CUDA DataLoader optimisations (only if running on CUDA):
    # - pin_memory speeds host-to-device transfer
    # - num_workers enables background loading
    if use_cuda:
        # NOTE: do not set shuffle=True for test loader
        train_kwargs.update({"num_workers": 1, "pin_memory": True})
        test_kwargs.update({"num_workers": 1, "pin_memory": True})

    train_transform = mnist.data_prep.transforms.build_train_augmentation(
        enabled=enable_train_augmentation,
        rotation_degrees=train_rotation_degrees,
        rotation_prob=train_rotation_prob,
        affine_prob=train_affine_prob,
        affine_translate_x=train_affine_translate_x,
        affine_translate_y=train_affine_translate_y,
        affine_scale_min=train_affine_scale_min,
        affine_scale_max=train_affine_scale_max,
        affine_shear_degrees=train_affine_shear_degrees,
        perspective_prob=train_perspective_prob,
        perspective_distortion_scale=train_perspective_distortion_scale,
        invert_prob=train_invert_prob,
    )

    # Build datasets from your processed MNIST PNG dataset layout.
    # Augmentation is applied at training time so each epoch sees new variants.
    train_dataset = mnist.data_prep.datasets.MNISTDataset(
        args["data_dir_path"],
        "train.csv",
        to_grayscale=True,
        to_tensor=True,
        transform=train_transform,
    )
    test_dataset = mnist.data_prep.datasets.MNISTDataset(
        args["data_dir_path"], "test.csv", to_grayscale=True, to_tensor=True
    )

    # Wrap datasets in DataLoaders (batches).
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # Model creation.
    model = mnist.modeling.models.Net().to(device)

    # Optimiser selection based on config.
    # Note: momentum only matters for SGD.
    if opt_name == "adam":
        optimiser = torch.optim.Adam(
            model.parameters(), lr=float(args["lr"]), weight_decay=weight_decay
        )
    elif opt_name == "sgd":
        optimiser = torch.optim.SGD(
            model.parameters(),
            lr=float(args["lr"]),
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif opt_name == "adadelta":
        optimiser = torch.optim.Adadelta(
            model.parameters(), lr=float(args["lr"]), weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}. Use: adam, sgd, adadelta")
    
    # Learning rate scheduler.
    # StepLR reduces LR each epoch by multiplying by gamma.
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=1, gamma=float(args["gamma"])
    )
    


    # Training loop.

    curr_test_loss = None
    curr_test_accuracy = None

    # step_offset supports resumed runs where you want epoch numbering to continue.
    for epoch in range(step_offset + 1, args["epochs"] + step_offset + 1):
        
        # One epoch of training.
        # Usually this function logs batch metrics and returns average train loss.
        curr_train_loss = mnist.modeling.utils.train(
            args, model, device, train_loader, optimiser, epoch, mlflow_init_status
        )

        # Evaluation on test set.
        # Returns average loss and accuracy for the epoch.
        curr_test_loss, curr_test_accuracy = mnist.modeling.utils.test(
            model, device, test_loader, epoch, mlflow_init_status
        )
        
        # Periodic checkpointing.
        # Saves model weights and optimiser state so you can resume.
        if epoch % args["model_checkpoint_interval"] == 0:
            logger.info("Exporting the model for epoch %s.", epoch)

            model_checkpoint_path = os.path.join(
                args["model_checkpoint_dir_path"], "checkpoint_model.pt"
            )
            os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "optimiser_state_dict": optimiser.state_dict(),
                    "train_loss": curr_train_loss,
                    "test_loss": curr_test_loss,
                },
                str(model_checkpoint_path),
            )

            # Log checkpoint to MLflow as an artifact.
            mnist.general_utils.mlflow_log(
                mlflow_init_status,
                "log_artifact",
                local_path=model_checkpoint_path,
                artifact_path="checkpoints",
                )
        # Update LR scheduler once per epoch.
        scheduler.step()

    # Log the training config file used.
    # Important: using get_original_cwd() ensures logging the real config, not Hydra run-dir copies.
    
    work_dir = Path(os.getcwd())
    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    final_path = data_dir / "model.pth"
    torch.save(model.state_dict(), str(final_path))
    logger.info("Saved final model state_dict to: %s", str(final_path))


    # Log the expected artifact path for CI to pull: data/model.pth
    mnist.general_utils.mlflow_log(
        mlflow_init_status,
        "log_artifact",
        local_path=str(final_path),
        artifact_path="data",
    )

    # -----------------------------
    # Log config + logs as artifacts
    # -----------------------------
    train_cfg_path = repo_root / "conf" / "train_model.yaml"
    mnist.general_utils.mlflow_log(
        mlflow_init_status,
        "log_artifact",
        local_path=str(train_cfg_path),
        artifact_path="config",
    )

    log_dir = args.get("log_dir", "logs")
    mnist.general_utils.mlflow_log(
        mlflow_init_status,
        "log_artifacts",
        local_dir=log_dir,
        artifact_path="logs",
    )

    # -----------------------------
    # Optional: log MLflow-native model + register
    # -----------------------------
    # This is useful for MLflow registry workflows, but your inference pipeline currently
    # consumes data/model.pth from GCS. Keeping both is OK as long as names are aligned.
    registered_model_name = args.get("registered_model_name", "mnist-net")

    mnist.general_utils.mlflow_pytorch_call(
        mlflow_init_status=mlflow_init_status,
        pytorch_function="log_model",
        pytorch_model=model,
        name="model",
        registered_model_name=registered_model_name,
    )

    # -----------------------------
    # Close MLflow run cleanly
    # -----------------------------
    if mlflow_init_status and mlflow_run is not None:
        artifact_uri = mlflow.get_artifact_uri()
        logger.info("Artifact URI: %s", artifact_uri)

        mnist.general_utils.mlflow_log(
            mlflow_init_status,
            "log_params",
            params={"artifact_uri": artifact_uri},
        )

        logger.info(
            "Training completed. REAL_MLFLOW_RUN_ID=%s",
            mlflow_run.info.run_id,
        )
        mlflow.end_run()
    else:
        logger.info("Training completed (MLflow disabled or failed).")

    # Return values used by Hydra Optuna sweeper if enabled.
    return curr_test_loss, curr_test_accuracy


if __name__ == "__main__":
    main()



    # train_cfg_path = os.path.join(hydra.utils.get_original_cwd(), "conf", "train_model.yaml")

    # mnist.general_utils.mlflow_log(
    #     mlflow_init_status,
    #     "log_artifact",
    #     local_path=str(final_path),
    #     artifact_path="config",
    # )


    # # Log the entire logs directory as artifacts (useful for debugging runs later).
    # train_cfg_path = repo_root / "conf" / "train_model.yaml"
    # mnist.general_utils.mlflow_log(
    #     mlflow_init_status,
    #     "log_artifact",
    #     local_path=str(train_cfg_path),
    #     artifact_path="config",
    # )


    # Log the final trained model to MLflow (pytorch-specific logging).
    # # This enables model registry versioning and later loading.
    # mnist.general_utils.mlflow_pytorch_call(
    #     mlflow_init_status=mlflow_init_status,
    #     pytorch_function="log_model",
    #     pytorch_model=model,
    #     name="model",
    #     registered_model_name=args.get("registered_model_name", "mnist-net"),
    # )

    # # Close MLflow run cleanly.
    # if mlflow_init_status:
    #     ## Get artifact link
    #     artifact_uri = mlflow.get_artifact_uri()
    #     logger.info("Artifact URI: %s", artifact_uri)

    #     # Log artifact URI as a param (handy for copying paths).
    #     mnist.general_utils.mlflow_log(
    #         mlflow_init_status, "log_params", params={"artifact_uri": artifact_uri}
    #     )
    #     logger.info(
    #         "Model training with MLflow run ID %s has completed.",
    #         mlflow_run.info.run_id,
    #     )
    #     mlflow.end_run()
    # else:
    #     logger.info("Model training has completed.")

    # # Return values are used by Hydra Optuna sweeper to optimise objectives.
    # # - first objective: minimize test loss
    # # - second objective: maximize accuracy
    # return curr_test_loss, curr_test_accuracy
