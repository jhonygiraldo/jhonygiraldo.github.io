#!/usr/bin/env python3
"""Cut the figures of the DL4CV Lecture 1 web note out of the compiled slides.

Most figures in this deck are TikZ drawn over real photographs and model
outputs, so they are exported as WebP (photo-friendly, small) rather than SVG.
Each entry below names a slide page and a rough crop box in page fractions
(x0, y0, x1, y1); the box is then trimmed to its non-white content, so the
boxes only need to be approximately right.

    python3 scripts/figures/dl4cv_lecture_01_figures.py PATH/TO/main.pdf
    python3 scripts/figures/dl4cv_lecture_01_figures.py PATH/TO/main.pdf --only fpn ious

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
    # Course introduction and computer-vision recap
    "beyond-a-single-label": (17, (0.14, 0.14, 0.93, 0.71)),
    "encode-decode": (20, (0.15, 0.22, 0.93, 0.44)),
    "inverse-problem": (21, (0.14, 0.14, 0.99, 0.82)),
    "why-hard": (23, (0.13, 0.28, 0.58, 0.78)),
    "progressive-annotations": (25, (0.13, 0.25, 0.93, 0.66)),
    "history": (26, (0.14, 0.20, 0.93, 0.66)),
    "hog-vs-cnn": (27, (0.14, 0.14, 0.99, 0.79)),
    "why-deep-learning": (28, (0.18, 0.14, 0.86, 0.52)),
    "classification": (30, (0.15, 0.18, 0.99, 0.80)),
    "pipeline": (31, (0.14, 0.1, 0.94, 0.57)),
    # CNN recap
    "convolution-intuition": (32, (0.14, 0.105, 0.94, 0.575)),
    "convolution-layer": (33, (0.15, 0.105, 0.99, 0.7)),
    "stride-padding-dilation": (34, (0.13, 0.30, 0.56, 0.71)),
    "pooling": (35, (0.15, 0.19, 0.93, 0.55)),
    "receptive-field": (36, (0.13, 0.16, 0.575, 0.935)),
    "cnn-pattern": (37, (0.14, 0.11, 0.9, 0.56)),
    "vgg": (39, (0.13, 0.18, 0.585, 0.76)),
    "resnet": (40, (0.13, 0.22, 0.57, 0.75)),
    "activations": (41, (0.57, 0.15, 0.99, 0.46)),
    "feature-hierarchy": (43, (0.14, 0.105, 0.94, 0.59)),
    "cnn-limitations": (45, (0.13, 0.2, 0.595, 0.86)),
    # Vision Transformer recap
    "attention-intuition": (46, (0.14, 0.24, 0.61, 0.94)),
    "patchify": (47, (0.14, 0.09, 0.93, 0.715)),
    "vit-pipeline": (48, (0.14, 0.15, 0.93, 0.73)),
    "qkv": (49, (0.14, 0.14, 0.93, 0.835)),
    "multi-head-attention": (50, (0.15, 0.14, 0.99, 0.6)),
    "transformer-block": (51, (0.13, 0.11, 0.58, 0.935)),
    "swin": (53, (0.14, 0.11, 0.99, 0.61)),
    "backbone": (54, (0.15, 0.14, 0.99, 0.5)),
    "fpn": (55, (0.13, 0.145, 0.93, 0.78)),
    # Object detection
    "localization-vs-detection": (61, (0.14, 0.14, 0.94, 0.76)),
    "box-parameterization": (62, (0.13, 0.16, 0.585, 0.75)),
    "detection-multitask": (64, (0.13, 0.19, 0.57, 0.8)),
    "iou": (65, (0.13, 0.17, 0.585, 0.8)),
    "precision-recall": (66, (0.13, 0.14, 0.57, 0.76)),
    "average-precision": (67, (0.13, 0.19, 0.555, 0.815)),
    "sliding-window": (68, (0.13, 0.17, 0.63, 0.94)),
    "fully-convolutional-detection": (69, (0.13, 0.14, 0.64, 0.63)),
    "anchors": (70, (0.13, 0.19, 0.55, 0.8)),
    "anchor-matching": (71, (0.13, 0.19, 0.585, 0.81)),
    "two-stage": (72, (0.15, 0.17, 0.99, 0.46)),
    "faster-rcnn": (73, (0.14, 0.1, 0.99, 0.66)),
    "one-stage": (74, (0.14, 0.11, 0.99, 0.565)),
    "yolo": (75, (0.13, 0.24, 0.585, 0.71)),
    "focal-loss": (76, (0.13, 0.165, 0.94, 0.86)),
    "nms": (77, (0.13, 0.155, 0.62, 0.8)),
    "nms-limitations": (78, (0.13, 0.17, 0.585, 0.79)),
    "set-prediction": (79, (0.13, 0.21, 0.6, 0.76)),
    "detr": (80, (0.14, 0.14, 0.99, 0.555)),
    "hungarian": (81, (0.13, 0.17, 0.55, 0.75)),
    "detection-failures": (83, (0.13, 0.21, 0.55, 0.66)),
    "boxes-vs-masks": (84, (0.14, 0.16, 0.92, 0.78)),
    # Segmentation
    "semantic-segmentation": (86, (0.14, 0.24, 0.56, 0.76)),
    "detection-vs-segmentation": (87, (0.14, 0.11, 0.99, 0.61)),
    "patch-classification": (88, (0.14, 0.24, 0.54, 0.76)),
    "fcn": (89, (0.14, 0.11, 0.99, 0.5)),
    "encoder-decoder": (90, (0.14, 0.17, 0.99, 0.53)),
    "upsampling": (91, (0.14, 0.14, 0.99, 0.53)),
    "skip-connections": (92, (0.13, 0.14, 0.61, 0.86)),
    "unet": (93, (0.13, 0.21, 0.625, 0.8)),
    "deeplab": (94, (0.13, 0.11, 0.99, 0.555)),
    "segmentation-losses": (95, (0.55, 0.19, 0.99, 0.76)),
    "segmentation-metrics": (96, (0.13, 0.24, 0.5, 0.76)),
    "instance-segmentation": (97, (0.14, 0.19, 0.61, 0.81)),
    "mask-rcnn": (98, (0.14, 0.11, 0.99, 0.47)),
    # Pose estimation and tracking
    "pose": (100, (0.14, 0.24, 0.56, 0.7)),
    "top-down-pose": (101, (0.14, 0.11, 0.99, 0.49)),
    "bottom-up-pose": (102, (0.14, 0.098, 0.99, 0.44)),
    "heatmaps": (103, (0.13, 0.11, 0.515, 0.8)),
    "pose-failures": (105, (0.14, 0.24, 0.565, 0.76)),
    "multi-object-tracking": (106, (0.14, 0.24, 0.61, 0.7)),
    "tracking-by-detection": (107, (0.14, 0.09, 0.93, 0.745)),
    "association": (108, (0.14, 0.11, 0.99, 0.545)),
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
