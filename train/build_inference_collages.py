import argparse
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_and_resize(image_path: Path, target_size: tuple[int, int], is_mask: bool = False) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if image.size != target_size:
        resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
        image = image.resize(target_size, resample)
    return image


def build_collages(input_dir: Path, output_dir: Path, collage_dir: Path) -> None:
    collage_dir.mkdir(parents=True, exist_ok=True)

    input_images = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )

    if not input_images:
        print(f"No input images found in {input_dir}")
        return

    built = 0
    skipped = 0

    for input_image in input_images:
        stem = input_image.stem
        mask_path = output_dir / f"{stem}_mask.png"
        attention_path = output_dir / f"{stem}_mask_attention_combined.png"

        if not mask_path.exists():
            skipped += 1
            continue

        input_rgb = Image.open(input_image).convert("RGB")
        target_size = input_rgb.size

        panels = [
            input_rgb,
            _load_and_resize(mask_path, target_size, is_mask=True),
        ]

        if attention_path.exists():
            panels.append(_load_and_resize(attention_path, target_size))

        gap = 12
        width = sum(img.width for img in panels) + gap * (len(panels) - 1)
        height = max(img.height for img in panels)

        collage = Image.new("RGB", (width, height), (18, 18, 18))

        x = 0
        for panel in panels:
            y = (height - panel.height) // 2
            collage.paste(panel, (x, y))
            x += panel.width + gap

        collage_path = collage_dir / f"{stem}_collage.png"
        collage.save(collage_path)
        built += 1

    print(f"Collages built: {built}")
    if skipped:
        print(f"Skipped (missing mask): {skipped}")
    print(f"Collage directory: {collage_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build side-by-side collages: input | mask | attention")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with original input images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory with inference outputs")
    parser.add_argument("--collage-dir", type=Path, required=True, help="Directory to save collages")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {args.input_dir}")
    if not args.output_dir.is_dir():
        raise SystemExit(f"Output directory not found: {args.output_dir}")

    build_collages(args.input_dir, args.output_dir, args.collage_dir)


if __name__ == "__main__":
    main()
