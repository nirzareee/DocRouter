"""Turn a clean PDF into one that looks scanned.

This is what creates the second arm of the experiment. Same document, same
ground truth, one variable changed: whether the text layer exists and how
clean the glyphs are.

The degraded output has NO text layer. Every backend that relies on extracting
embedded text (`naive`, `pylib`) will return nothing, which is the point:
the router must learn that these documents require an OCR-capable backend, and
the benchmark must show the cliff.

Pipeline: render each page to a bitmap, apply skew + blur + noise + JPEG
artifacts, then reassemble as an image-only PDF.

    python scripts/degrade.py in.pdf out.pdf --severity medium
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageFilter

# Severity presets. Keep these named and fixed: "medium" must mean the same
# thing in every run, or your degraded split is not a controlled condition.
PRESETS = {
    "light":  {"dpi": 200, "skew": 0.3, "blur": 0.4, "noise": 4,  "jpeg": 85, "contrast": 0.97},
    "medium": {"dpi": 150, "skew": 0.8, "blur": 0.8, "noise": 10, "jpeg": 60, "contrast": 0.90},
    "heavy":  {"dpi": 110, "skew": 1.6, "blur": 1.3, "noise": 18, "jpeg": 40, "contrast": 0.82},
}


def degrade_image(img: Image.Image, cfg: dict, rng: np.random.Generator) -> Image.Image:
    img = img.convert("L")

    # Skew: pages fed through a scanner are never perfectly square.
    angle = rng.uniform(-cfg["skew"], cfg["skew"])
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)

    if cfg["blur"] > 0:
        img = img.filter(ImageFilter.GaussianBlur(cfg["blur"]))

    arr = np.asarray(img).astype(np.float32)

    # Contrast reduction: photocopied text is grey, not black.
    arr = 255 - (255 - arr) * cfg["contrast"]

    # Sensor noise.
    arr += rng.normal(0, cfg["noise"], arr.shape)

    # Gentle illumination gradient, as from a flatbed lid not fully closed.
    h, w = arr.shape
    gradient = np.linspace(0.97, 1.03, w)[None, :] * np.linspace(1.02, 0.98, h)[:, None]
    arr *= gradient

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # JPEG artifacts, applied last so the ringing lands on the degraded glyphs.
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=cfg["jpeg"])
    buf.seek(0)
    return Image.open(buf).convert("L")


def degrade_pdf(src: Path, dst: Path, severity: str = "medium", seed: int = 0) -> int:
    cfg = PRESETS[severity]
    # Seeded so the degraded corpus is reproducible. An unseeded corpus cannot
    # be regenerated, which makes every number in your results table a
    # one-time observation.
    rng = np.random.default_rng(seed)

    pdf = pdfium.PdfDocument(str(src))
    scale = cfg["dpi"] / 72.0
    pages = []
    for i in range(len(pdf)):
        bitmap = pdf[i].render(scale=scale)
        pages.append(degrade_image(bitmap.to_pil(), cfg, rng))
    pdf.close()

    if not pages:
        raise ValueError(f"No pages rendered from {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    pages[0].convert("RGB").save(
        dst, format="PDF", save_all=True,
        append_images=[p.convert("RGB") for p in pages[1:]],
        resolution=cfg["dpi"],
    )
    return len(pages)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--severity", choices=sorted(PRESETS), default="medium")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n = degrade_pdf(Path(args.src), Path(args.dst), args.severity, args.seed)
    print(f"Wrote {args.dst} ({n} pages, severity={args.severity}, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
