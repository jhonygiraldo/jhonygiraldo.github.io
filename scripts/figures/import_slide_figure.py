#!/usr/bin/env python3
"""Import a figure from the LaTeX slides into assets/figures/.

The beamer sources keep every figure as a vector PDF. `pdftocairo -svg` turns
one into an SVG whose text is already converted to paths, so it renders the
same everywhere, but it writes coordinates at full double precision. Rounding
them to two decimals is visually lossless at these sizes and cuts the file by
roughly a third.

Figures that embed a raster (a photograph, a heat map, a dense scatter) do not
benefit from that and are better served as PNG; pass --png for those.

    python3 scripts/figures/import_slide_figure.py SOURCE.pdf out-name
    python3 scripts/figures/import_slide_figure.py SOURCE.pdf out-name --png --dpi 200
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "figures"

NUMBER = re.compile(r"-?\d+\.\d+")


def _round(match: re.Match) -> str:
    text = f"{round(float(match.group()), 2):.2f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-") else "0"


def shrink_svg(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    # Keep the XML declaration intact: rounding would turn version="1.0" into "1".
    head, separator, body = text.partition("?>")
    body = NUMBER.sub(_round, body)
    body = re.sub(r'(<svg[^>]*?\sversion=")1(")', r"\g<1>1.1\g<2>", body, count=1)
    body = re.sub(r"\n\s*\n", "\n", body)
    destination.write_text(head + separator + body, encoding="utf-8")


def quantize_png(path: Path) -> None:
    """Palette-reduce a flat-colour line drawing; roughly halves the file."""
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image.quantize(colors=256, method=Image.MEDIANCUT).save(path, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="the figure PDF inside the beamer project")
    parser.add_argument("name", help="output basename under assets/figures/")
    parser.add_argument("--png", action="store_true", help="rasterize instead of keeping it vector")
    parser.add_argument("--dpi", type=int, default=200, help="resolution for --png (default 200)")
    args = parser.parse_args()

    if not args.source.exists():
        print(f"error: {args.source} does not exist", file=sys.stderr)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    if args.png:
        destination = OUT / f"{args.name}.png"
        subprocess.run(
            ["pdftocairo", "-png", "-r", str(args.dpi), "-singlefile", str(args.source), str(OUT / args.name)],
            check=True,
        )
        quantize_png(destination)
    else:
        destination = OUT / f"{args.name}.svg"
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            subprocess.run(["pdftocairo", "-svg", str(args.source), str(temporary)], check=True)
            shrink_svg(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    print(f"{destination.relative_to(ROOT)}  {destination.stat().st_size / 1024:.0f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
