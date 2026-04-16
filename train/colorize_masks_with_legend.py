import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


BASE_COLORS: List[Tuple[int, int, int]] = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (57, 59, 121),
    (82, 84, 163),
    (107, 110, 207),
    (156, 158, 222),
    (99, 121, 57),
    (140, 162, 82),
    (181, 207, 107),
    (206, 219, 156),
    (140, 109, 49),
    (189, 158, 57),
]


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans.ttf",
        "arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_text(draw: ImageDraw.ImageDraw, position: Tuple[int, int], text: str, font: ImageFont.ImageFont, fill: Tuple[int, int, int]) -> None:
    try:
        draw.text(position, text, fill=fill, font=font)
    except UnicodeEncodeError:
        fallback_text = text.encode("latin-1", errors="replace").decode("latin-1")
        draw.text(position, fallback_text, fill=fill, font=font)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    try:
        if hasattr(draw, "textbbox"):
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            return right - left, bottom - top
        return draw.textsize(text, font=font)
    except UnicodeEncodeError:
        fallback_text = text.encode("latin-1", errors="replace").decode("latin-1")
        if hasattr(draw, "textbbox"):
            left, top, right, bottom = draw.textbbox((0, 0), fallback_text, font=font)
            return right - left, bottom - top
        return draw.textsize(fallback_text, font=font)


def _color_for_index(idx: int) -> Tuple[int, int, int]:
    if idx < len(BASE_COLORS):
        return BASE_COLORS[idx]
    r = (37 * idx + 91) % 256
    g = (71 * idx + 47) % 256
    b = (19 * idx + 173) % 256
    return int(r), int(g), int(b)


def _load_mask_values_from_checkpoint(checkpoint_path: Path) -> List[int]:
    if not checkpoint_path.exists():
        return []

    try:
        try:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        print(f"Warning: failed to load checkpoint for mask_values: {exc}")
        return []

    mask_values = state.get("mask_values") if isinstance(state, dict) else None
    if not isinstance(mask_values, list) or not mask_values:
        return []

    scalar_values: List[int] = []
    for value in mask_values:
        if isinstance(value, list):
            return []
        try:
            scalar_values.append(int(value))
        except (TypeError, ValueError):
            return []

    return scalar_values


def _resolve_class_values(mask_paths: Sequence[Path], classes: int, checkpoint_path: Optional[Path]) -> List[int]:
    if checkpoint_path is not None:
        values = _load_mask_values_from_checkpoint(checkpoint_path)
        if values:
            return values

    if classes > 0:
        return list(range(classes))

    unique_values = set()
    for mask_path in mask_paths:
        mask_np = np.array(Image.open(mask_path))
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]
        unique_values.update(np.unique(mask_np).tolist())

    return sorted(int(v) for v in unique_values)


def _build_legend_image(
    class_labels: Sequence[str],
    class_colors: Sequence[Tuple[int, int, int]],
) -> Image.Image:
    labels = list(class_labels)
    title = "Class Legend"
    font = _load_font(size=28)
    title_font = _load_font(size=32)

    padding = 26
    row_height = 48
    swatch_size = 32

    draw_probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    label_width = 0
    for label in labels:
        text_w, _ = _text_size(draw_probe, label, font)
        label_width = max(label_width, text_w)

    title_w, title_h = _text_size(draw_probe, title, title_font)
    panel_width = max(520, padding * 3 + swatch_size + label_width, padding * 2 + title_w)
    panel_height = padding * 3 + title_h + len(labels) * row_height

    panel = Image.new("RGB", (panel_width, panel_height), (24, 24, 24))
    panel_draw = ImageDraw.Draw(panel)
    _draw_text(panel_draw, (padding, padding), title, title_font, (245, 245, 245))

    start_y = padding * 2 + title_h + 8
    for idx, (label, color) in enumerate(zip(labels, class_colors)):
        y = start_y + idx * row_height
        panel_draw.rectangle(
            (padding, y + 6, padding + swatch_size, y + 6 + swatch_size),
            fill=color,
            outline=(220, 220, 220),
            width=2,
        )
        _draw_text(panel_draw, (padding + swatch_size + 14, y + 4), label, font, (245, 245, 245))

    return panel


def _load_class_name_map(class_names_path: Optional[Path]) -> Dict[str, str]:
    if class_names_path is None:
        return {}
    if not class_names_path.exists():
        print(f"Warning: class names file not found: {class_names_path}")
        return {}

    try:
        with class_names_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"Warning: failed to parse class names file: {exc}")
        return {}

    if not isinstance(data, dict):
        print("Warning: class names file must contain a JSON object")
        return {}

    output: Dict[str, str] = {}
    for key, value in data.items():
        if isinstance(value, str):
            output[str(key)] = value
    return output


def _legend_label_for(value: int, idx: int, class_name_map: Dict[str, str]) -> str:
    if str(value) in class_name_map:
        return f"{value}: {class_name_map[str(value)]}"
    if str(idx) in class_name_map:
        return f"{value}: {class_name_map[str(idx)]}"
    return f"Class {value}"


def colorize_masks(
    mask_dir: Path,
    output_dir: Path,
    classes: int,
    checkpoint_path: Optional[Path],
    class_names_path: Optional[Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(mask_dir.glob("*_mask.png"))
    if not mask_paths:
        print(f"No mask files found in: {mask_dir}")
        return

    class_values = _resolve_class_values(mask_paths, classes, checkpoint_path)
    if not class_values:
        print("No class values resolved. Nothing to colorize.")
        return

    class_name_map = _load_class_name_map(class_names_path)

    class_to_color = {value: _color_for_index(idx) for idx, value in enumerate(class_values)}
    legend_values = list(class_values)
    legend_colors = [class_to_color[v] for v in legend_values]
    legend_labels = [_legend_label_for(v, idx, class_name_map) for idx, v in enumerate(legend_values)]

    created = 0
    for mask_path in mask_paths:
        mask_np = np.array(Image.open(mask_path))
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]

        color_np = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
        for value, color in class_to_color.items():
            color_np[mask_np == value] = color

        unknown_values = sorted(set(np.unique(mask_np).tolist()) - set(class_to_color.keys()))
        if unknown_values:
            for value in unknown_values:
                color = _color_for_index(len(legend_values))
                class_to_color[value] = color
                legend_values.append(value)
                legend_colors.append(color)
                legend_labels.append(f"Class {value}")
                color_np[mask_np == value] = color

        color_img = Image.fromarray(color_np, mode="RGB")

        stem = mask_path.stem
        out_name = stem.replace("_mask", "_mask_color") + ".png"
        color_img.save(output_dir / out_name)
        created += 1

    legend_img = _build_legend_image(legend_labels, legend_colors)
    legend_img.save(output_dir / "legend.png")

    print(f"Colorized masks created: {created}")
    print("Legend image created: legend.png")
    print(f"Output directory: {output_dir}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create colorized segmentation masks with class legend")
    parser.add_argument("--mask-dir", type=Path, required=True, help="Directory containing *_mask.png files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save colorized outputs")
    parser.add_argument("--classes", type=int, default=0, help="Class count for legend fallback (0 = auto)")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint path to read mask_values")
    parser.add_argument("--class-names", type=Path, default=None, help="Optional JSON map for class names")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if not args.mask_dir.is_dir():
        raise SystemExit(f"Mask directory not found: {args.mask_dir}")
    colorize_masks(args.mask_dir, args.output_dir, args.classes, args.checkpoint, args.class_names)
