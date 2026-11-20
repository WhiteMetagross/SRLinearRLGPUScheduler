"""Generate and package the approved architecture Figures 1, 2, and 3.

The supplied AI generated PNG files remain immutable visual references. Figure
1 and Figure 3 receive deterministic factual corrections before packaging.
Figure 2 remains unchanged. PNG outputs are byte exact copies of the corrected
sources. SVG outputs embed those pixels without resampling. PDF outputs place
the same pixels on a page with the matching aspect ratio.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "paper" / "figure_sources"
REFERENCE_DIR = SOURCE_DIR / "references"
OUT_DIR = ROOT / "paper" / "figures"
PDF_DIR = OUT_DIR / "pdf"
SVG_DIR = OUT_DIR / "svg"
PNG_DIR = OUT_DIR / "png"
MANIFEST_PATH = OUT_DIR / "architecture_source_manifest.json"

SOURCE_MAP = {
    "fig01_srlinear_architecture": "Fig1.png",
    "fig02_srac_system_architecture": "Fig2.png",
    "fig03_srac_rl_agent_architecture": "Fig3.png",
}

PDF_DPI = 300


SERIF = Path(font_manager.findfont("DejaVu Serif"))
SERIF_BOLD = Path(
    font_manager.findfont(font_manager.FontProperties(family="DejaVu Serif", weight="bold"))
)
SANS = Path(font_manager.findfont("DejaVu Sans"))
SANS_BOLD = Path(
    font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
)


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def _center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    *,
    fill: tuple[int, int, int] = (14, 18, 24),
    y: int | None = None,
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = left + (right - left - width) / 2 - bounds[0]
    if y is None:
        y = int(top + (bottom - top - height) / 2 - bounds[1])
    draw.text((x, y), text, font=font, fill=fill)


def _rounded_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    outline: tuple[int, int, int],
    radius: int = 18,
    width: int = 4,
) -> None:
    draw.rounded_rectangle(
        box,
        radius=radius,
        fill=(253, 253, 253),
        outline=outline,
        width=width,
    )


def _arrow_head(
    draw: ImageDraw.ImageDraw,
    tip: tuple[int, int],
    *,
    direction: str,
    fill: tuple[int, int, int],
    size: int = 13,
) -> None:
    x, y = tip
    if direction == "left":
        points = [(x, y), (x + size, y - size // 2), (x + size, y + size // 2)]
    elif direction == "right":
        points = [(x, y), (x - size, y - size // 2), (x - size, y + size // 2)]
    elif direction == "down":
        points = [(x, y), (x - size // 2, y - size), (x + size // 2, y - size)]
    else:
        raise ValueError(f"Unsupported arrow direction: {direction}")
    draw.polygon(points, fill=fill)


def _vertical_background_strip(
    image: Image.Image,
    *,
    source_box: tuple[int, int, int, int],
    destination_box: tuple[int, int, int, int],
    clean_rows: tuple[tuple[int, int], ...] = (),
) -> None:
    patch = image.crop(source_box)
    if clean_rows:
        pixels = np.asarray(patch).copy()
        for start, end in clean_rows:
            start = max(1, int(start))
            end = min(int(end), pixels.shape[0] - 1)
            if end <= start:
                continue
            top = pixels[start - 1].astype(np.float64)
            bottom = pixels[end].astype(np.float64)
            for row in range(start, end):
                weight = (row - start + 1) / (end - start + 1)
                pixels[row] = np.rint((1.0 - weight) * top + weight * bottom).astype(np.uint8)
        patch = Image.fromarray(pixels, mode="RGB")
    width = destination_box[2] - destination_box[0]
    height = destination_box[3] - destination_box[1]
    image.paste(patch.resize((width, height), Image.Resampling.BICUBIC), destination_box)


def _render_corrected_fig01() -> None:
    reference = REFERENCE_DIR / "Fig1_ai_reference.png"
    destination = SOURCE_DIR / "Fig1.png"
    image = Image.open(reference).convert("RGB")
    draw = ImageDraw.Draw(image)

    black = (13, 16, 21)
    navy = (15, 38, 68)
    blue = (8, 92, 171)
    amber = (221, 151, 0)
    green = (3, 126, 69)

    # Correct the benchmark input cardinality and data semantics.
    draw.rounded_rectangle((72, 233, 345, 375), radius=12, fill=(253, 253, 253))
    _center_text(draw, (72, 241, 345, 286), "Request History", _font(SERIF_BOLD, 30), fill=black)
    _center_text(draw, (72, 291, 345, 326), "28 organization", _font(SERIF, 25), fill=black)
    _center_text(draw, (72, 328, 345, 363), "request series", _font(SERIF, 25), fill=black)

    # Show all five causal seasonal anchors without expanding the panel count.
    draw.rounded_rectangle((446, 243, 716, 371), radius=12, fill=(253, 253, 253))
    _center_text(draw, (446, 255, 716, 300), "Short Scale Anchors", _font(SERIF_BOLD, 22), fill=black)
    _center_text(draw, (446, 310, 716, 351), "24 and 48 hour lags", _font(SERIF, 22), fill=black)

    draw.rounded_rectangle((446, 454, 716, 594), radius=12, fill=(253, 253, 253))
    _center_text(draw, (446, 464, 716, 509), "Long Scale Anchors", _font(SERIF_BOLD, 22), fill=black)
    _center_text(draw, (446, 516, 716, 551), "168, 336, and 672", _font(SERIF, 22), fill=black)
    _center_text(draw, (446, 553, 716, 584), "hour lags", _font(SERIF, 22), fill=black)

    # Name the two residual branches precisely.
    draw.rounded_rectangle((812, 507, 1107, 585), radius=8, fill=(253, 253, 253))
    _center_text(draw, (812, 516, 1107, 546), "Organization and", _font(SERIF, 23), fill=black)
    _center_text(draw, (812, 550, 1107, 580), "direct pool", _font(SERIF, 23), fill=black)

    # Rebuild the coherent forecast panel so the two estimates are parallel.
    # A clean right margin from the immutable reference supplies the panel tint.
    _vertical_background_strip(
        image,
        source_box=(1533, 154, 1541, 739),
        destination_box=(1169, 154, 1549, 739),
        clean_rows=((238, 269),),
    )
    draw = ImageDraw.Draw(image)

    organization_box = (1184, 214, 1352, 366)
    pool_box = (1364, 214, 1532, 366)
    reconciliation_box = (1210, 535, 1506, 669)
    _rounded_card(draw, organization_box, outline=blue)
    _rounded_card(draw, pool_box, outline=amber)
    _rounded_card(draw, reconciliation_box, outline=green)

    _center_text(draw, organization_box, "Organization", _font(SERIF_BOLD, 21), fill=black, y=238)
    _center_text(draw, organization_box, "Path", _font(SERIF_BOLD, 21), fill=black, y=270)
    _center_text(draw, organization_box, "Seasonal plus", _font(SERIF, 17), fill=black, y=311)
    _center_text(draw, organization_box, "residual", _font(SERIF, 17), fill=black, y=336)
    _center_text(draw, pool_box, "Direct Pool", _font(SERIF_BOLD, 21), fill=black, y=238)
    _center_text(draw, pool_box, "Path", _font(SERIF_BOLD, 21), fill=black, y=270)
    _center_text(draw, pool_box, "Aggregate plus", _font(SERIF, 17), fill=black, y=311)
    _center_text(draw, pool_box, "residual", _font(SERIF, 17), fill=black, y=336)
    _center_text(draw, reconciliation_box, "Coherent Reconciliation", _font(SERIF_BOLD, 21), fill=black, y=562)
    _center_text(draw, reconciliation_box, "Pool blend", _font(SERIF, 18), fill=black, y=606)
    _center_text(draw, reconciliation_box, "Exact organization hierarchy", _font(SERIF, 18), fill=black, y=634)

    # Both paths join before reconciliation. There is no sequential dependency.
    organization_x = (organization_box[0] + organization_box[2]) // 2
    pool_x = (pool_box[0] + pool_box[2]) // 2
    merge_y = 481
    merge_x = (organization_x + pool_x) // 2
    draw.line((organization_x, organization_box[3], organization_x, merge_y), fill=navy, width=4)
    draw.line((pool_x, pool_box[3], pool_x, merge_y), fill=navy, width=4)
    draw.line((organization_x, merge_y, pool_x, merge_y), fill=navy, width=4)
    draw.line((merge_x, merge_y, merge_x, reconciliation_box[1]), fill=navy, width=4)
    draw.ellipse((merge_x - 5, merge_y - 5, merge_x + 5, merge_y + 5), fill=navy)
    _arrow_head(draw, (merge_x, reconciliation_box[1]), direction="down", fill=navy)
    parallel_label = (1290, 414, 1426, 452)
    draw.rounded_rectangle(parallel_label, radius=9, fill=(246, 250, 252))
    _center_text(
        draw,
        parallel_label,
        "Parallel estimates",
        _font(SERIF_BOLD, 15),
        fill=navy,
    )

    # Restore the inter panel arrows hidden by the panel reconstruction.
    draw.line((1142, 406, 1181, 406), fill=navy, width=4)
    _arrow_head(draw, (1181, 406), direction="right", fill=navy)
    draw.line((1506, 602, 1541, 602), fill=navy, width=4)
    draw.line((1541, 602, 1541, 406), fill=navy, width=4)
    draw.line((1541, 406, 1580, 406), fill=navy, width=4)
    _arrow_head(draw, (1580, 406), direction="right", fill=navy)

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)


def _render_corrected_fig03() -> None:
    reference = REFERENCE_DIR / "Fig3_ai_reference.png"
    destination = SOURCE_DIR / "Fig3.png"
    image = Image.open(reference).convert("RGB")
    draw = ImageDraw.Draw(image)

    black = (13, 16, 21)
    dark_gray = (64, 80, 94)
    blue = (6, 105, 190)

    # State is intentionally summarized, but every executed feature family is named.
    state_box = (54, 168, 292, 390)
    _rounded_card(draw, state_box, outline=dark_gray, radius=18, width=5)
    _center_text(draw, state_box, "Selected State", _font(SANS_BOLD, 24), fill=black, y=195)
    state_font = _font(SANS, 18)
    for y, text in zip(
        (242, 270, 298, 326, 354),
        (
            "Capacity and queues",
            "Quota endpoints",
            "Forecast and uncertainty",
            "Previous actions",
            "Costs and budgets",
        ),
    ):
        _center_text(draw, state_box, text, state_font, fill=dark_gray, y=y)

    # Name the stochastic actor and the four distinct value heads precisely.
    actor_box = (812, 165, 1095, 389)
    _rounded_card(draw, actor_box, outline=blue, radius=18, width=5)
    _center_text(draw, actor_box, "Actor and Critics", _font(SANS_BOLD, 24), fill=black, y=205)
    _center_text(draw, actor_box, "Tanh Gaussian actor", _font(SANS, 18), fill=dark_gray, y=254)
    _center_text(draw, actor_box, "Controls: alpha, beta", _font(SANS, 18), fill=dark_gray, y=288)
    _center_text(draw, actor_box, "One reward critic", _font(SANS, 18), fill=dark_gray, y=322)
    _center_text(draw, actor_box, "Three constraint critics", _font(SANS, 18), fill=dark_gray, y=356)

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)


def _render_corrected_fig02() -> None:
    reference = REFERENCE_DIR / "Fig2_ai_reference.png"
    destination = SOURCE_DIR / "Fig2.png"
    image = Image.open(reference).convert("RGB")

    # Replace the broad original headers with precise control and execution roles.
    for source_box, destination_box in (
        ((694, 80, 704, 128), (694, 80, 1028, 128)),
        ((1068, 80, 1078, 128), (1068, 80, 1410, 128)),
    ):
        patch = image.crop(source_box)
        image.paste(
            patch.resize(
                (
                    destination_box[2] - destination_box[0],
                    destination_box[3] - destination_box[1],
                ),
                Image.Resampling.BICUBIC,
            ),
            destination_box,
        )
    draw = ImageDraw.Draw(image)

    black = (13, 16, 21)
    white = (250, 252, 255)
    _center_text(
        draw,
        (686, 70, 1035, 139),
        "SRAC Quota Control",
        _font(SERIF_BOLD, 28),
        fill=white,
        y=85,
    )
    _center_text(
        draw,
        (1062, 70, 1416, 139),
        "GFS Scheduling Engine",
        _font(SERIF_BOLD, 24),
        fill=white,
        y=88,
    )

    label_box = (1084, 502, 1405, 551)
    draw.rounded_rectangle(label_box, radius=10, fill=(253, 253, 253))
    _center_text(
        draw,
        label_box,
        "Priority Aware Preemption",
        _font(SERIF_BOLD, 22),
        fill=black,
    )

    # Use the canonical name for the deterministic quota endpoint.
    endpoint_box = (714, 231, 1010, 276)
    draw.rounded_rectangle(endpoint_box, radius=8, fill=(253, 253, 253))
    _center_text(
        draw,
        endpoint_box,
        "CLQ and SRLinear",
        _font(SERIF, 22),
        fill=black,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)


def _render_corrected_sources() -> None:
    _render_corrected_fig01()
    _render_corrected_fig02()
    _render_corrected_fig03()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_svg(source: Path, destination: Path, width: int, height: int) -> None:
    encoded = base64.b64encode(source.read_bytes()).decode("ascii")
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'  <image x="0" y="0" width="{width}" height="{height}" '
        f'preserveAspectRatio="none" '
        f'href="data:image/png;base64,{encoded}" '
        f'xlink:href="data:image/png;base64,{encoded}"/>\n'
        '</svg>\n'
    )
    destination.write_text(svg, encoding="utf-8", newline="\n")


def _write_pdf(source: Path, destination: Path, width: int, height: int) -> None:
    pixels = np.asarray(Image.open(source).convert("RGB"))
    figure = plt.figure(
        figsize=(width / PDF_DPI, height / PDF_DPI),
        dpi=PDF_DPI,
        frameon=False,
    )
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axes.imshow(pixels, interpolation="none", aspect="auto")
    axes.set_xlim(-0.5, width - 0.5)
    axes.set_ylim(height - 0.5, -0.5)
    axes.axis("off")
    figure.savefig(
        destination,
        format="pdf",
        dpi=PDF_DPI,
        bbox_inches=None,
        pad_inches=0.0,
        facecolor="white",
    )
    plt.close(figure)


def _write_manifest() -> None:
    records = {}
    for stem, source_name in SOURCE_MAP.items():
        source = SOURCE_DIR / source_name
        if not source.exists():
            continue
        with Image.open(source) as image:
            width, height = image.size
            mode = image.mode
        records[stem] = {
            "source": str(source.relative_to(ROOT)),
            "source_sha256": _sha256(source),
            "width_pixels": width,
            "height_pixels": height,
            "mode": mode,
            "png_output": str((PNG_DIR / f"{stem}.png").relative_to(ROOT)),
            "svg_output": str((SVG_DIR / f"{stem}.svg").relative_to(ROOT)),
            "pdf_output": str((PDF_DIR / f"{stem}.pdf").relative_to(ROOT)),
            "svg_representation": "lossless embedded PNG",
            "pdf_representation": "full page embedded raster",
            "pdf_dpi": PDF_DPI,
        }
    MANIFEST_PATH.write_text(
        json.dumps({"schema_version": "1.0", "figures": records}, indent=2) + "\n",
        encoding="utf-8",
    )


def _package(stem: str) -> None:
    source_name = SOURCE_MAP[stem]
    source = SOURCE_DIR / source_name
    if not source.is_file():
        raise FileNotFoundError(f"Approved figure source is missing: {source}")
    for directory in (PDF_DIR, SVG_DIR, PNG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        width, height = image.size
        if image.mode != "RGB":
            raise ValueError(f"Approved figure must remain RGB: {source}")

    png_output = PNG_DIR / f"{stem}.png"
    shutil.copyfile(source, png_output)
    if _sha256(source) != _sha256(png_output):
        raise AssertionError(f"PNG output is not byte exact: {png_output}")

    _write_svg(source, SVG_DIR / f"{stem}.svg", width, height)
    _write_pdf(source, PDF_DIR / f"{stem}.pdf", width, height)
    _write_manifest()
    print(f"[OK] Packaged {stem}: {width} x {height} pixels")


def generate_fig01_srlinear_architecture() -> None:
    _package("fig01_srlinear_architecture")


def generate_fig02_srac_system_architecture() -> None:
    _package("fig02_srac_system_architecture")


def generate_fig03_srac_rl_agent_architecture() -> None:
    _package("fig03_srac_rl_agent_architecture")


def main() -> None:
    _render_corrected_sources()
    generate_fig01_srlinear_architecture()
    generate_fig02_srac_system_architecture()
    generate_fig03_srac_rl_agent_architecture()


if __name__ == "__main__":
    main()
