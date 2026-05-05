import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support


LABELS = ["56", "57", "58", "59", "60"]


def labels_from_confusion_matrix(matrix):
    y_true = []
    y_pred = []
    for true_index, row in enumerate(matrix):
        for pred_index, count in enumerate(row):
            y_true.extend([true_index] * count)
            y_pred.extend([pred_index] * count)
    return np.array(y_true), np.array(y_pred)


def main():
    parser = argparse.ArgumentParser(description="Create summary files from an existing metrics.json.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/thai_handwriting_56_60_torch"))
    args = parser.parse_args()

    metrics_path = args.model_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    labels = [str(label) for label in metrics.get("labels", LABELS)]
    y_true, y_pred = labels_from_confusion_matrix(matrix)

    report_text = classification_report(y_true, y_pred, target_names=labels, digits=4)
    report_dict = classification_report(y_true, y_pred, target_names=labels, digits=4, output_dict=True)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics.update(
        {
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall),
            "weighted_f1": float(weighted_f1),
            "classification_report": report_dict,
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.model_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    accuracy = float(metrics.get("accuracy", report_dict["accuracy"]))
    summary_text = "\n".join(
        [
            "Training Summary",
            "================",
            f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)",
            f"Macro precision: {macro_precision:.4f}",
            f"Macro recall: {macro_recall:.4f}",
            f"Macro F1-score: {macro_f1:.4f}",
            f"Weighted precision: {weighted_precision:.4f}",
            f"Weighted recall: {weighted_recall:.4f}",
            f"Weighted F1-score: {weighted_f1:.4f}",
            "",
            f"Train samples: {metrics.get('train_count', '-')}",
            f"Validation samples: {metrics.get('validation_count', '-')}",
            f"Test samples: {metrics.get('test_count', '-')}",
            "",
            "Confusion matrix",
            str(matrix),
            "",
            "Classification report",
            report_text,
        ]
    )
    summary_path = args.model_dir / "training_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Updated: {metrics_path}")
    print(f"Updated: {summary_path}")


if __name__ == "__main__":
    main()
