#!/usr/bin/env python3
"""Cut the figures of the DL4CV Lecture 1 web note out of the compiled slides.

Most figures in this deck are TikZ drawn over real photographs and model
outputs, so they are exported as WebP (photo-friendly, small) rather than SVG.
Each entry below names a slide page and a rough crop box in page fractions
(x0, y0, x1, y1); the box is then trimmed to its non-white content, so the
boxes only need to be approximately right.

    python3 scripts/figures/dl4cv_lecture_01_figures.py PATH/TO/main.pdf
    python3 scripts/figures/dl4cv_lecture_01_figures.py PATH/TO/main.pdf --only fpn iou

Requires pdftoppm (poppler) and Pillow with WebP support.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "figures" / "dl4cv"
PREFIX = "l01-"
DPI = 300
PAD = 10  # pixels of white margin kept around the trimmed content

# name: (page, (x0, y0, x1, y1))
FIGURES: dict[str, tuple[int, tuple[float, float, float, float]]] = {
    # Computer-vision recap
    "beyond-a-single-label": (17, (0.14, 0.08, 0.93, 0.61)),
    "encode-decode": (21, (0.15, 0.22, 0.93, 0.46)),
    "inverse-problem": (22, (0.13, 0.20, 0.635, 0.80)),
    "why-hard": (24, (0.13, 0.25, 0.58, 0.7)),
    "progressive-annotations": (26, (0.13, 0.25, 0.93, 0.66)),
    "history": (27, (0.14, 0.20, 0.93, 0.66)),
    "hog-vs-cnn": (28, (0.14, 0.10, 0.99, 0.665)),
    "why-deep-learning": (29, (0.18, 0.14, 0.86, 0.52)),
    "pipeline": (32, (0.14, 0.075, 0.94, 0.465)),
    # CNN recap
    "convolution-intuition": (34, (0.14, 0.085, 0.94, 0.49)),
    "convolution-layer": (35, (0.15, 0.09, 0.99, 0.59)),
    "stride-padding-dilation": (36, (0.13, 0.25, 0.53, 0.64)),
    "pooling": (37, (0.15, 0.19, 0.93, 0.56)),
    "receptive-field": (38, (0.13, 0.14, 0.575, 0.705)),
    "cnn-pattern": (39, (0.14, 0.11, 0.90, 0.56)),
    "vgg": (40, (0.13, 0.18, 0.585, 0.435)),
    "resnet": (41, (0.13, 0.4, 0.56, 0.88)),
    "activations": (42, (0.57, 0.15, 0.99, 0.42)),
    "feature-hierarchy": (44, (0.14, 0.105, 0.94, 0.59)),
    "fpn": (45, (0.13, 0.115, 0.93, 0.7)),
    "cnn-limitations": (47, (0.13, 0.20, 0.595, 0.72)),
    # Vision Transformer recap
    "attention-intuition": (49, (0.14, 0.135, 0.61, 0.66)),
    "patchify": (50, (0.14, 0.09, 0.93, 0.64)),
    "vit-pipeline": (51, (0.14, 0.105, 0.93, 0.68)),
    "qkv": (52, (0.14, 0.14, 0.93, 0.745)),
    "multi-head-attention": (53, (0.15, 0.09, 0.99, 0.53)),
    "transformer-block": (54, (0.13, 0.11, 0.58, 0.79)),
    "swin": (55, (0.14, 0.1, 0.99, 0.51)),
    "backbone": (56, (0.15, 0.14, 0.99, 0.55)),
    # Object detection
    "localization-vs-detection": (60, (0.14, 0.14, 0.94, 0.655)),
    "box-parameterization": (61, (0.13, 0.105, 0.585, 0.69)),
    "detection-multitask": (63, (0.13, 0.125, 0.57, 0.73)),
    "iou": (64, (0.13, 0.17, 0.585, 0.83)),
    "precision-recall": (65, (0.13, 0.1, 0.57, 0.72)),
    "sliding-window": (66, (0.13, 0.17, 0.63, 0.94)),
    "two-stage": (67, (0.15, 0.25, 0.99, 0.48)),
    "region-proposals": (69, (0.14, 0.11, 0.99, 0.52)),
    "nms": (70, (0.13, 0.14, 0.6, 0.8)),
    "nms-limitations": (71, (0.13, 0.27, 0.585, 0.58)),
    "rcnn": (72, (0.14, 0.11, 0.99, 0.58)),
    "fast-rcnn": (73, (0.14, 0.085, 0.99, 0.42)),
    "anchors": (74, (0.13, 0.19, 0.55, 0.69)),
    "anchor-matching": (75, (0.13, 0.19, 0.585, 0.77)),
    "faster-rcnn": (76, (0.14, 0.10, 0.99, 0.60)),
    "rcnn-to-faster-rcnn": (77, (0.14, 0.10, 0.93, 0.70)),
    "one-stage": (78, (0.14, 0.11, 0.99, 0.565)),
    "yolo": (79, (0.13, 0.24, 0.585, 0.71)),
    "class-imbalance": (80, (0.15, 0.33, 0.92, 0.80)),
    "focal-motivation": (81, (0.14, 0.295, 0.99, 0.63)),
    "focal-loss": (82, (0.13, 0.125, 0.94, 0.735)),
    "set-prediction": (84, (0.13, 0.21, 0.60, 0.685)),
    "detr": (85, (0.14, 0.095, 0.99, 0.52)),
    "hungarian": (86, (0.13, 0.17, 0.55, 0.75)),
    # Segmentation
    "semantic-segmentation": (88, (0.14, 0.125, 0.56, 0.57)),
    "patch-classification": (89, (0.14, 0.24, 0.54, 0.76)),
    "fcn": (90, (0.14, 0.1, 0.99, 0.41)),
    "encoder-decoder": (91, (0.14, 0.155, 0.99, 0.46)),
    "upsampling": (92, (0.14, 0.14, 0.99, 0.53)),
    "skip-connections": (93, (0.13, 0.12, 0.61, 0.77)),
    "unet": (94, (0.13, 0.18, 0.625, 0.73)),
    "deeplab": (95, (0.13, 0.11, 0.99, 0.455)),
    "segmentation-metrics": (96, (0.13, 0.21, 0.52, 0.59)),
    "instance-segmentation": (97, (0.14, 0.19, 0.61, 0.60)),
    "mask-rcnn": (98, (0.14, 0.13, 0.99, 0.4)),
    # Pose estimation and tracking
    "pose": (100, (0.14, 0.22, 0.56, 0.635)),
    "keypoint-rcnn": (101, (0.14, 0.10, 0.99, 0.635)),
    "multi-object-tracking": (102, (0.14, 0.18, 0.61, 0.62)),
    "tracking-by-detection": (103, (0.14, 0.085, 0.93, 0.82)),
}


def trim(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, (255, 255, 255))
    difference = ImageChops.difference(rgb, background).convert("L").point(lambda v: 255 if v > 24 else 0)
    box = difference.getbbox()
    if box is None:
        return rgb
    x0, y0, x1, y1 = box
    return rgb.crop((max(x0 - PAD, 0), max(y0 - PAD, 0), min(x1 + PAD, rgb.width), min(y1 + PAD, rgb.height)))


def render_page(pdf: Path, page: int, directory: Path) -> Image.Image:
    stem = directory / f"page-{page:03d}"
    subprocess.run(
        ["pdftoppm", "-r", str(DPI), "-png", "-f", str(page), "-l", str(page), "-singlefile", str(pdf), str(stem)],
        check=True,
    )
    return Image.open(f"{stem}.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path, help="compiled Lecture 1 slides")
    parser.add_argument("--only", nargs="*", help="export only these figure names")
    args = parser.parse_args()
    if not args.pdf.exists():
        print(f"error: {args.pdf} does not exist", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    selected = {name: spec for name, spec in FIGURES.items() if not args.only or name in args.only}
    with tempfile.TemporaryDirectory() as temporary:
        cache: dict[int, Image.Image] = {}
        for name, (page, (x0, y0, x1, y1)) in selected.items():
            if page not in cache:
                cache[page] = render_page(args.pdf, page, Path(temporary))
            image = cache[page]
            width, height = image.size
            crop = image.crop((round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)))
            destination = OUT / f"{PREFIX}{name}.webp"
            trim(crop).save(destination, "WEBP", quality=86, method=6)
            print(f"{destination.relative_to(ROOT)}  (slide {page})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
