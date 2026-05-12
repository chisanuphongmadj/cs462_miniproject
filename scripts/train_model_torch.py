import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

try:
    from scripts.image_preprocess import random_affine_on_canvas, split_digit_images
except ModuleNotFoundError:
    from image_preprocess import random_affine_on_canvas, split_digit_images


LABELS = ["56", "57", "58", "59", "60"]
DIGIT_LABELS = ["0", "5", "6", "7", "8", "9"]
DIGIT_TO_INDEX = {label: index for index, label in enumerate(DIGIT_LABELS)}


class DigitCropDataset(Dataset):
    def __init__(self, items, image_size, augment=False):
        self.items = items
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, digit_index, side = self.items[index]
        image = Image.open(path).convert("L")
        left_digit, right_digit = split_digit_images(image, self.image_size)
        digit_image = left_digit if side == "left" else right_digit

        if self.augment:
            digit_image = random_affine_on_canvas(digit_image, random)

        tensor = image_to_tensor(digit_image)
        return tensor, torch.tensor(digit_index, dtype=torch.long)


class SmallCnn(nn.Module):
    def __init__(self, class_count):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(96, class_count),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def image_to_tensor(image):
    array = np.asarray(image, dtype=np.float32)
    array = (255.0 - array) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def collect_split_paths(data_dir, seed):
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}

    for class_index, label in enumerate(LABELS):
        class_dir = data_dir / label
        paths = sorted(class_dir.glob("*.png"))
        if len(paths) < 10:
            raise ValueError(f"Need at least 10 images for class {label}, found {len(paths)}")

        rng.shuffle(paths)
        train_count = int(len(paths) * 0.70)
        val_count = int(len(paths) * 0.15)
        split["train"].extend((path, class_index) for path in paths[:train_count])
        split["val"].extend((path, class_index) for path in paths[train_count:train_count + val_count])
        split["test"].extend((path, class_index) for path in paths[train_count + val_count:])

    for items in split.values():
        rng.shuffle(items)
    return split


def make_digit_items(number_items):
    digit_items = []
    for path, class_index in number_items:
        label = LABELS[class_index]
        digit_items.append((path, DIGIT_TO_INDEX[label[0]], "left"))
        digit_items.append((path, DIGIT_TO_INDEX[label[1]], "right"))
    return digit_items


def make_class_weights(digit_items):
    counts = np.zeros(len(DIGIT_LABELS), dtype=np.float32)
    for _, digit_index, _ in digit_items:
        counts[digit_index] += 1
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(DIGIT_LABELS) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def valid_label_probabilities(left_probs, right_probs):
    scores = []
    for label in LABELS:
        left_index = DIGIT_TO_INDEX[label[0]]
        right_index = DIGIT_TO_INDEX[label[1]]
        scores.append(float(left_probs[left_index] * right_probs[right_index]))

    scores = np.asarray(scores, dtype=np.float32)
    total = float(scores.sum())
    if total <= 0:
        return np.full(len(LABELS), 1.0 / len(LABELS), dtype=np.float32)
    return scores / total


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    seen = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        seen += x.size(0)

    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate_digits(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    predictions = []
    targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        correct += (pred == y).sum().item()
        seen += x.size(0)
        predictions.extend(pred.cpu().numpy().tolist())
        targets.extend(y.cpu().numpy().tolist())

    return total_loss / seen, correct / seen, np.array(targets), np.array(predictions)


@torch.no_grad()
def evaluate_numbers(model, number_items, image_size, device):
    model.eval()
    targets = []
    predictions = []
    confidences = []

    for path, class_index in number_items:
        image = Image.open(path).convert("L")
        left_digit, right_digit = split_digit_images(image, image_size)
        batch = torch.stack([image_to_tensor(left_digit), image_to_tensor(right_digit)]).to(device)
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        label_probs = valid_label_probabilities(probs[0], probs[1])
        prediction = int(label_probs.argmax())

        targets.append(class_index)
        predictions.append(prediction)
        confidences.append(float(label_probs[prediction]))

    return np.array(targets), np.array(predictions), np.array(confidences)


def plot_history(history, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_acc"], label="train")
    axes[0].plot(history["val_acc"], label="val")
    axes[0].set_title("Digit Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_loss"], label="train")
    axes[1].plot(history["val_loss"], label="val")
    axes[1].set_title("Digit Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train digit-pair PyTorch CNN for Thai handwriting labels 56-60.")
    parser.add_argument("--data", type=Path, default=Path("dataset/thai_handwriting_56_60"))
    parser.add_argument("--out", type=Path, default=Path("models/thai_handwriting_56_60_torch"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    args.out.mkdir(parents=True, exist_ok=True)
    split_paths = collect_split_paths(args.data, args.seed)
    train_digit_items = make_digit_items(split_paths["train"])
    val_digit_items = make_digit_items(split_paths["val"])
    test_digit_items = make_digit_items(split_paths["test"])

    train_loader = DataLoader(
        DigitCropDataset(train_digit_items, args.image_size, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        DigitCropDataset(val_digit_items, args.image_size),
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        DigitCropDataset(test_digit_items, args.image_size),
        batch_size=args.batch_size,
    )

    model = SmallCnn(len(DIGIT_LABELS)).to(device)
    class_weights = make_class_weights(train_digit_items).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    patience = 10
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_digits(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"digit_train_loss={train_loss:.4f} digit_train_acc={train_acc:.4f} "
            f"digit_val_loss={val_loss:.4f} digit_val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print("Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    digit_test_loss, digit_test_acc, digit_y_test, digit_y_pred = evaluate_digits(model, test_loader, criterion, device)
    y_test, y_pred, number_confidences = evaluate_numbers(model, split_paths["test"], args.image_size, device)

    report_text = classification_report(y_test, y_pred, target_names=LABELS, digits=4)
    report_dict = classification_report(y_test, y_pred, target_names=LABELS, digits=4, output_dict=True)
    digit_report_text = classification_report(
        digit_y_test,
        digit_y_pred,
        labels=list(range(len(DIGIT_LABELS))),
        target_names=DIGIT_LABELS,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    model_path = args.out / "thai_handwriting_56_60.pt"
    torch.save(
        {
            "model_type": "digit_pair",
            "model_state_dict": model.state_dict(),
            "labels": LABELS,
            "digit_labels": DIGIT_LABELS,
            "image_size": args.image_size,
            "test_accuracy": float(accuracy),
            "digit_test_accuracy": float(digit_test_acc),
            "preprocess": "split_digit_images_v1",
        },
        model_path,
    )
    (args.out / "labels.json").write_text(json.dumps(LABELS, indent=2), encoding="utf-8")
    (args.out / "digit_labels.json").write_text(json.dumps(DIGIT_LABELS, indent=2), encoding="utf-8")
    (args.out / "classification_report.txt").write_text(report_text, encoding="utf-8")
    summary_text = "\n".join(
        [
            "Training Summary",
            "================",
            "Model type: digit_pair",
            f"Number accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)",
            f"Digit accuracy: {digit_test_acc:.4f} ({digit_test_acc * 100:.2f}%)",
            f"Macro precision: {macro_precision:.4f}",
            f"Macro recall: {macro_recall:.4f}",
            f"Macro F1-score: {macro_f1:.4f}",
            f"Weighted precision: {weighted_precision:.4f}",
            f"Weighted recall: {weighted_recall:.4f}",
            f"Weighted F1-score: {weighted_f1:.4f}",
            "",
            f"Train number samples: {len(split_paths['train'])}",
            f"Validation number samples: {len(split_paths['val'])}",
            f"Test number samples: {len(split_paths['test'])}",
            f"Train digit samples: {len(train_digit_items)}",
            f"Validation digit samples: {len(val_digit_items)}",
            f"Test digit samples: {len(test_digit_items)}",
            f"Mean valid-label confidence: {float(number_confidences.mean()):.4f}",
            "",
            "Number confusion matrix",
            str(matrix),
            "",
            "Number classification report",
            report_text,
            "",
            "Digit classification report",
            digit_report_text,
        ]
    )
    (args.out / "training_summary.txt").write_text(summary_text, encoding="utf-8")
    (args.out / "metrics.json").write_text(
        json.dumps(
            {
                "model_type": "digit_pair",
                "accuracy": float(accuracy),
                "digit_accuracy": float(digit_test_acc),
                "macro_precision": float(macro_precision),
                "macro_recall": float(macro_recall),
                "macro_f1": float(macro_f1),
                "weighted_precision": float(weighted_precision),
                "weighted_recall": float(weighted_recall),
                "weighted_f1": float(weighted_f1),
                "digit_test_loss": float(digit_test_loss),
                "labels": LABELS,
                "digit_labels": DIGIT_LABELS,
                "train_count": len(split_paths["train"]),
                "validation_count": len(split_paths["val"]),
                "test_count": len(split_paths["test"]),
                "train_digit_count": len(train_digit_items),
                "validation_digit_count": len(val_digit_items),
                "test_digit_count": len(test_digit_items),
                "confusion_matrix": matrix.tolist(),
                "classification_report": report_dict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_history(history, args.out / "training_history.png")

    print("\nNumber classification report")
    print(report_text)
    print("Number confusion matrix")
    print(matrix)
    print("\nDigit classification report")
    print(digit_report_text)
    print(f"\nSaved model: {model_path}")
    print(f"Saved summary: {args.out / 'training_summary.txt'}")


if __name__ == "__main__":
    main()
