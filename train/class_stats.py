# python train/class_stats.py --masks-dir train/data/masks --num-classes 68 --ignore-index 255 --output-csv train/results_stats/class_stats.csv --output-plot train/results_stats/class_stats.png

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-class statistics for segmentation masks."
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=Path("train/data/masks"),
        help="Directory with mask images (default: train/data/masks)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Total number of classes. If omitted, inferred from masks.",
    )
    parser.add_argument(
        "--ignore-index",
        type=int,
        action="append",
        default=[],
        help="Class id to ignore (can be passed multiple times).",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".png"],
        help="Mask file extensions to include (default: .png).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path to save statistics.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help="Optional image path for bar chart visualization (.png/.jpg).",
    )
    parser.add_argument(
        "--plot-top-k",
        type=int,
        default=20,
        help="Show top K classes by pixel count in the plot (default: 20).",
    )
    return parser.parse_args()


def collect_mask_paths(masks_dir: Path, extensions: Sequence[str]) -> List[Path]:
    normalized_ext = {
        ext if ext.startswith(".") else f".{ext}"
        for ext in (item.lower() for item in extensions)
    }
    files = [
        p
        for p in sorted(masks_dir.iterdir())
        if p.is_file() and p.suffix.lower() in normalized_ext
    ]
    return files


def validate_and_read_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path))
    if mask.ndim != 2:
        raise ValueError(
            f"Mask must be index map (H, W), got shape {mask.shape} in {path}"
        )
    return mask


def compute_stats(
    mask_paths: Sequence[Path], ignore_indices: Iterable[int]
) -> Tuple[Counter, Counter, int]:
    ignore_set = set(ignore_indices)
    pixel_counts: Counter = Counter()
    image_counts: Counter = Counter()
    non_empty_masks = 0

    for mask_path in mask_paths:
        mask = validate_and_read_mask(mask_path)
        if ignore_set:
            keep = ~np.isin(mask, list(ignore_set))
            mask = mask[keep]

        if mask.size == 0:
            continue

        non_empty_masks += 1
        unique, counts = np.unique(mask, return_counts=True)
        for class_id, count in zip(unique.tolist(), counts.tolist()):
            pixel_counts[int(class_id)] += int(count)
            image_counts[int(class_id)] += 1

    return pixel_counts, image_counts, non_empty_masks


def build_rows(
    pixel_counts: Counter,
    image_counts: Counter,
    class_ids: Sequence[int],
    total_pixels: int,
    total_images: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for class_id in class_ids:
        pixels = int(pixel_counts.get(class_id, 0))
        images = int(image_counts.get(class_id, 0))
        pixel_percent = (100.0 * pixels / total_pixels) if total_pixels else 0.0
        image_percent = (100.0 * images / total_images) if total_images else 0.0
        rows.append(
            {
                "class_id": class_id,
                "pixel_count": pixels,
                "pixel_percent": pixel_percent,
                "image_count": images,
                "image_percent": image_percent,
            }
        )
    return rows


def print_rows(rows: Sequence[Dict[str, float]]) -> None:
    print("class_id | pixel_count | pixel_% | image_count | image_%")
    print("-" * 58)
    for row in rows:
        print(
            f"{int(row['class_id']):8d} | "
            f"{int(row['pixel_count']):11d} | "
            f"{row['pixel_percent']:7.3f} | "
            f"{int(row['image_count']):11d} | "
            f"{row['image_percent']:7.3f}"
        )


def write_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "pixel_count",
                "pixel_percent",
                "image_count",
                "image_percent",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_plot(path: Path, rows: Sequence[Dict[str, float]], top_k: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    plot_rows = [row for row in rows if int(row["pixel_count"]) > 0]
    if not plot_rows:
        plot_rows = list(rows)

    plot_rows = sorted(plot_rows, key=lambda row: int(row["pixel_count"]), reverse=True)
    if top_k > 0:
        plot_rows = plot_rows[:top_k]

    labels = [str(int(row["class_id"])) for row in plot_rows]
    pixel_values = [int(row["pixel_count"]) for row in plot_rows]
    image_values = [int(row["image_count"]) for row in plot_rows]
    y_pos = np.arange(len(labels))

    fig_width = max(10.0, len(labels) * 0.45)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 6), constrained_layout=True)

    axes[0].barh(y_pos, pixel_values, color="#2A9D8F")
    axes[0].set_title("Pixels per class")
    axes[0].set_xlabel("Pixels")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels)
    axes[0].set_ylabel("Class ID")
    axes[0].invert_yaxis()

    axes[1].barh(y_pos, image_values, color="#E76F51")
    axes[1].set_title("Images containing class")
    axes[1].set_xlabel("Images")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()

    fig.suptitle("Class frequency overview", fontsize=13)

    ensure_parent_dir(path)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def ensure_parent_dir(path: Path) -> None:
    parent = path.parent
    if parent.exists() and not parent.is_dir():
        raise NotADirectoryError(
            f"Output parent path is not a directory: {parent}. "
            "Choose a different output path or rename/remove this file."
        )
    parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    masks_dir = args.masks_dir
    if not masks_dir.exists() or not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    mask_paths = collect_mask_paths(masks_dir, args.extensions)
    if not mask_paths:
        raise RuntimeError(
            f"No mask files found in {masks_dir} with extensions: {args.extensions}"
        )

    pixel_counts, image_counts, used_images = compute_stats(mask_paths, args.ignore_index)

    if args.num_classes is not None:
        out_of_range = [class_id for class_id in pixel_counts if class_id >= args.num_classes]
        if out_of_range:
            raise ValueError(
                "Found class ids >= num_classes: "
                f"{sorted(out_of_range)} (num_classes={args.num_classes})"
            )
        class_ids = list(range(args.num_classes))
    else:
        class_ids = sorted(pixel_counts.keys())

    total_pixels = int(sum(pixel_counts.get(class_id, 0) for class_id in class_ids))
    rows = build_rows(pixel_counts, image_counts, class_ids, total_pixels, used_images)

    print(f"Masks scanned: {len(mask_paths)}")
    print(f"Masks used after ignore filter: {used_images}")
    print(f"Total counted pixels: {total_pixels}")
    print_rows(rows)

    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
        print(f"\nSaved CSV: {args.output_csv}")

    if args.output_plot is not None:
        write_plot(args.output_plot, rows, args.plot_top_k)
        print(f"Saved plot: {args.output_plot}")


if __name__ == "__main__":
    main()
