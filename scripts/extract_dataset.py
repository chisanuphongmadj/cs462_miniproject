import argparse
import base64
import csv
import json
from pathlib import Path


def decode_data_url(data_url):
    if "," not in data_url:
        raise ValueError("imageData is not a data URL")
    header, encoded = data_url.split(",", 1)
    if "image/png" not in header:
        raise ValueError(f"expected PNG data URL, got: {header}")
    return base64.b64decode(encoded)


def main():
    parser = argparse.ArgumentParser(
        description="Extract Thai handwriting dataset JSON into class folders."
    )
    parser.add_argument("json_path", type=Path, help="Path to exported dataset JSON")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dataset/thai_handwriting_56_60"),
        help="Output folder for extracted PNG files",
    )
    args = parser.parse_args()

    with args.json_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    samples = dataset.get("samples", [])
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.csv"
    counts = {}

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["filename", "label", "thai_label", "created_at"])

        for index, sample in enumerate(samples, start=1):
            label = str(sample["label"])
            thai_label = sample.get("thaiLabel", label)
            class_dir = args.out / label
            class_dir.mkdir(parents=True, exist_ok=True)

            counts[label] = counts.get(label, 0) + 1
            filename = f"{label}_{counts[label]:04d}.png"
            output_path = class_dir / filename
            output_path.write_bytes(decode_data_url(sample["imageData"]))

            writer.writerow(
                [
                    str(output_path.relative_to(args.out)).replace("\\", "/"),
                    label,
                    thai_label,
                    sample.get("createdAt", ""),
                ]
            )

    print(f"Extracted {len(samples)} images to {args.out}")
    for label in sorted(counts):
        print(f"{label}: {counts[label]}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
