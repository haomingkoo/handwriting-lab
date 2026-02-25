"""Evaluate a trained MNIST model and export metrics for the web report page."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

# Allow `import mnist` when running script from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import mnist  # noqa: E402


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="data/local-model-export/model.pth",
        help="Path to model checkpoint/state_dict.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed/mnist-pngs-data-aisg-processed",
        help="Path to processed MNIST data with train.csv/test.csv.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Which split CSV to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--use-mps",
        action="store_true",
        help="Enable MPS if available.",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Enable CUDA if available.",
    )
    parser.add_argument(
        "--output-json",
        default="reports/evaluation_latest.json",
        help="Where to save evaluation results JSON.",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_json = Path(args.output_json).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    anno_file_name = f"{args.split}.csv"
    dataset = mnist.data_prep.datasets.MNISTDataset(
        data_dir_path=str(data_dir),
        anno_file_name=anno_file_name,
        to_grayscale=True,
        to_tensor=True,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
    )

    model, device = mnist.modeling.utils.load_model(
        path_to_model=str(model_path),
        use_cuda=bool(args.use_cuda),
        use_mps=bool(args.use_mps),
    )
    model.eval()

    confusion = torch.zeros((10, 10), dtype=torch.int64)
    total = 0
    correct = 0

    with torch.no_grad():
        for _, data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            preds = output.argmax(dim=1)

            total += int(target.numel())
            correct += int((preds == target).sum().item())

            for t, p in zip(target.view(-1), preds.view(-1)):
                confusion[int(t.item()), int(p.item())] += 1

    per_class_metrics: list[dict[str, float | int]] = []
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for label in range(10):
        tp = int(confusion[label, label].item())
        fp = int(confusion[:, label].sum().item() - tp)
        fn = int(confusion[label, :].sum().item() - tp)
        support = int(confusion[label, :].sum().item())

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        per_class_metrics.append(
            {
                "label": label,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": support,
            }
        )

    macro_precision /= 10
    macro_recall /= 10
    macro_f1 /= 10

    off_diag_pairs: list[dict[str, int]] = []
    for actual in range(10):
        for predicted in range(10):
            if actual == predicted:
                continue
            count = int(confusion[actual, predicted].item())
            if count > 0:
                off_diag_pairs.append(
                    {"actual": actual, "predicted": predicted, "count": count}
                )
    off_diag_pairs.sort(key=lambda item: item["count"], reverse=True)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "split": args.split,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "accuracy": round(_safe_div(correct, total), 6),
        "num_samples": total,
        "num_correct": correct,
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion.tolist(),
        "top_misclassifications": off_diag_pairs[:20],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to: {output_json}")
    print(
        "Accuracy: "
        f"{payload['accuracy']:.4f} ({payload['num_correct']}/{payload['num_samples']})"
    )


if __name__ == "__main__":
    main()
