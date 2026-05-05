import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


LABELS = ["56", "57", "58", "59", "60"]


def load_image(path, image_size):
    image = Image.open(path).convert("L")
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32)
    # Canvas exports black strokes on white background. Invert so strokes are high values.
    array = (255.0 - array) / 255.0
    return np.expand_dims(array, axis=-1)


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
        train_paths = paths[:train_count]
        val_paths = paths[train_count:train_count + val_count]
        test_paths = paths[train_count + val_count:]

        split["train"].extend((path, class_index) for path in train_paths)
        split["val"].extend((path, class_index) for path in val_paths)
        split["test"].extend((path, class_index) for path in test_paths)

    for items in split.values():
        rng.shuffle(items)

    return split


def load_split(items, image_size):
    x = np.stack([load_image(path, image_size) for path, _ in items])
    y = np.asarray([label for _, label in items], dtype=np.int64)
    return x, y


def build_model(image_size, class_count):
    inputs = tf.keras.Input(shape=(image_size, image_size, 1))
    x = tf.keras.layers.RandomRotation(0.08)(inputs)
    x = tf.keras.layers.RandomTranslation(0.08, 0.08)(x)
    x = tf.keras.layers.RandomZoom(0.10)(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(96, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train CNN for Thai handwriting labels 56-60.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset/thai_handwriting_56_60"),
        help="Dataset folder with class subfolders 56, 57, 58, 59, 60",
    )
    parser.add_argument("--out", type=Path, default=Path("models/thai_handwriting_56_60"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)

    split_paths = collect_split_paths(args.data, args.seed)
    x_train, y_train = load_split(split_paths["train"], args.image_size)
    x_val, y_val = load_split(split_paths["val"], args.image_size)
    x_test, y_test = load_split(split_paths["test"], args.image_size)

    print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")
    print(f"Image shape: {x_train.shape[1:]}")

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(len(x_train), seed=args.seed)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(args.batch_size)

    model = build_model(args.image_size, len(LABELS))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=0.00001,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    probabilities = model.predict(x_test, batch_size=args.batch_size)
    y_pred = np.argmax(probabilities, axis=1)

    report_text = classification_report(y_test, y_pred, target_names=LABELS, digits=4)
    matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    model_path = args.out / "thai_handwriting_56_60.keras"
    model.save(model_path)

    labels_path = args.out / "labels.json"
    labels_path.write_text(json.dumps(LABELS, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = {
        "accuracy": float(accuracy),
        "labels": LABELS,
        "train_count": int(len(x_train)),
        "validation_count": int(len(x_val)),
        "test_count": int(len(x_test)),
        "confusion_matrix": matrix.tolist(),
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.out / "classification_report.txt").write_text(report_text, encoding="utf-8")
    plot_history(history, args.out / "training_history.png")

    print("\nClassification report")
    print(report_text)
    print("Confusion matrix")
    print(matrix)
    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
