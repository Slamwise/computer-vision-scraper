"""Build a YOLO training set from saved eBay HTML, with labels taken from the DOM.

The original project hand-labelled screenshots for darknet. Here labels are
free: we render archived/cached result pages offline (``capture.render_html``),
ask the DOM where every card and field is drawn (``capture.card_regions``), and
screenshot the page as fixed-height tiles with known page offsets
(``capture.capture_tiles``). Converting each region box into each tile's pixel
space gives exact YOLO labels for every tile.

Rendering one page at several viewport widths (and optionally scale factors)
is the main augmentation: eBay reflows its cards with the window width, which
is exactly the variation real screenshots have.

The train/val split is made per source page, never per tile: neighbouring
tiles of one page overlap and share styling, so splitting tiles would leak
near-duplicates into validation and inflate the scores.
"""

from __future__ import annotations

import gzip
import json
import logging
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from pydantic import BaseModel, Field

from ..config import BrowserSettings
from ..models import VISION_CLASSES, Box, CardRegion

log = logging.getLogger(__name__)

# Minimum share of a box that must fall inside a tile for it to be labelled there.
# Text fields are single lines (~15-20 CSS px) and the tile overlap (160 px) is far
# taller, so every field is whole in some tile: a line cut in half is unreadable, and
# labelling it would only teach the detector to fire on fragments OCR cannot read.
# Cards (~150-300 px) can be taller than the overlap, so some are never whole in any
# tile; labelling the visible part keeps the detector finding them, and the page
# extractor stitches the pieces back together. Under ~30 % the visible strip is
# mostly padding with no readable field. A product photo is recognisable from a
# part, so it gets the same lenient threshold as the card.
MIN_VISIBLE: dict[str, float] = {"listing": 0.3, "image": 0.3}
MIN_VISIBLE_TEXT_FIELD = 0.5

CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(VISION_CLASSES)}


class DatasetSummary(BaseModel):
    out_dir: Path
    data_yaml: Path
    ground_truth: Path
    train_pages: list[str]
    val_pages: list[str]
    train_images: int = 0
    val_images: int = 0
    # split -> class name -> number of labelled boxes
    instances: dict[str, dict[str, int]] = Field(default_factory=dict)
    renders: int = 0  # (page, width, scale) combinations rendered


def read_html(path: str | Path) -> str:
    """Read a saved page, gzip-compressed (``.html.gz``) or not."""
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
            return f.read()
    return p.read_text(encoding="utf-8", errors="replace")


def page_stem(path: str | Path) -> str:
    name = Path(path).name
    for suffix in (".html.gz", ".htm.gz", ".html", ".htm", ".gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def min_visible_for(name: str, min_visible: float | Mapping[str, float] | None = None) -> float:
    if isinstance(min_visible, (int, float)):
        return float(min_visible)
    if min_visible and name in min_visible:
        return float(min_visible[name])
    return MIN_VISIBLE.get(name, MIN_VISIBLE_TEXT_FIELD)


def clip_box(box: Box, tile: Box) -> Box | None:
    """The part of ``box`` inside ``tile`` (same coordinate space), or ``None``."""
    x1, y1 = max(box.x, tile.x), max(box.y, tile.y)
    x2, y2 = min(box.x + box.w, tile.x + tile.w), min(box.y + box.h, tile.y + tile.h)
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(x=x1, y=y1, w=x2 - x1, h=y2 - y1)


def to_tile_pixels(box: Box, tile_box: Box, scale: float = 1.0) -> Box:
    """Page (CSS px) box -> pixel box in the tile's screenshot (device px)."""
    return Box(x=(box.x - tile_box.x) * scale, y=(box.y - tile_box.y) * scale, w=box.w * scale, h=box.h * scale)


def _region_boxes(region: CardRegion) -> Iterable[tuple[str, Box]]:
    yield "listing", region.box
    for name in VISION_CLASSES[1:]:
        if name in region.fields:
            yield name, region.fields[name]


def visible_boxes(regions: list[CardRegion], tile_box: Box, *,
                  min_visible: float | Mapping[str, float] | None = None) -> list[tuple[str, Box, float]]:
    """``(class, clipped page box, visible fraction)`` for every box shown enough in ``tile_box``."""
    out: list[tuple[str, Box, float]] = []
    for region in regions:
        for name, box in _region_boxes(region):
            if box.area <= 0:
                continue
            clipped = clip_box(box, tile_box)
            if clipped is None:
                continue
            frac = clipped.area / box.area
            if frac + 1e-9 < min_visible_for(name, min_visible):
                continue
            out.append((name, clipped, frac))
    return out


def regions_to_yolo(regions: list[CardRegion], tile_box: Box, img_w: int, img_h: int, *,
                    scale: float = 1.0, min_visible: float | Mapping[str, float] | None = None) -> list[str]:
    """YOLO label lines (``cls cx cy w h``, normalised to [0, 1]) for one tile.

    ``regions`` and ``tile_box`` are in page CSS pixels; ``img_w``/``img_h`` are
    the tile screenshot's size in device pixels (CSS px * ``scale``, the
    device scale factor). Boxes are clipped to the tile; boxes less visible
    than ``min_visible`` (default: ``MIN_VISIBLE`` / ``MIN_VISIBLE_TEXT_FIELD``)
    are dropped.
    """
    lines: list[str] = []
    for name, clipped, _ in visible_boxes(regions, tile_box, min_visible=min_visible):
        px = to_tile_pixels(clipped, tile_box, scale)
        x1, y1 = max(0.0, px.x), max(0.0, px.y)
        x2, y2 = min(float(img_w), px.x + px.w), min(float(img_h), px.y + px.h)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h
        w, h = (x2 - x1) / img_w, (y2 - y1) / img_h
        lines.append(f"{CLASS_INDEX[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def tile_ground_truth(regions: list[CardRegion], tile_box: Box, *, scale: float = 1.0,
                      min_visible: float | Mapping[str, float] | None = None) -> list[dict]:
    """Regions visible in a tile, in tile pixel coordinates, with their DOM texts."""
    out: list[dict] = []
    for region in regions:
        clipped = clip_box(region.box, tile_box)
        if clipped is None or clipped.area / max(region.box.area, 1e-9) < min_visible_for("listing", min_visible):
            continue
        fields: dict[str, dict] = {}
        for name, box in region.fields.items():
            c = clip_box(box, tile_box)
            if c is None or box.area <= 0:
                continue
            frac = c.area / box.area
            if frac + 1e-9 < min_visible_for(name, min_visible):
                continue
            fields[name] = {"box": to_tile_pixels(c, tile_box, scale).model_dump(), "visible": round(frac, 3)}
        out.append({
            "item_id": region.item_id,
            "box": to_tile_pixels(clipped, tile_box, scale).model_dump(),
            "visible": round(clipped.area / region.box.area, 3),
            "fields": fields,
            "texts": {k: v for k, v in region.texts.items() if k in fields},
        })
    return out


def split_sources(sources: list[Path], *, val_sources: list[Path] | None = None, val_fraction: float = 0.2,
                  seed: int = 0) -> tuple[list[Path], list[Path]]:
    """Split source pages into (train, val). Pages, not tiles, are the unit."""
    srcs = [Path(s) for s in sources]
    if val_sources is not None:
        val = [Path(v) for v in val_sources]
        held = {p.resolve() for p in val}
        return [s for s in srcs if s.resolve() not in held], val
    order = sorted(srcs, key=lambda p: p.name)
    random.Random(seed).shuffle(order)
    n_val = int(round(len(order) * val_fraction))
    if len(order) > 1:
        n_val = min(max(n_val, 1 if val_fraction > 0 else 0), len(order) - 1)
    else:
        n_val = 0
    val = order[:n_val]
    train = [s for s in srcs if s not in val]
    return train, val


def write_data_yaml(out_dir: Path, *, has_val: bool = True) -> Path:
    """Ultralytics dataset file; class names in ``VISION_CLASSES`` order."""
    lines = [
        "# Auto-labelled from rendered eBay HTML by ebay_sold.vision.dataset",
        f"path: {out_dir.resolve()}",
        "train: images/train",
        f"val: {'images/val' if has_val else 'images/train'}",
        f"nc: {len(VISION_CLASSES)}",
        "names:",
        *(f"  {i}: {name}" for i, name in enumerate(VISION_CLASSES)),
    ]
    path = out_dir / "data.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _png_size(path: Path) -> tuple[int, int]:
    # PNG IHDR: width and height are big-endian uint32 at bytes 16..24.
    with path.open("rb") as f:
        head = f.read(24)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


async def build_dataset(sources: list[Path], out_dir: Path, *, val_sources: list[Path] | None = None,
                        val_fraction: float = 0.2, viewport_widths: tuple[int, ...] = (1280, 1440),
                        device_scale_factors: tuple[float, ...] = (1.0,), tile_height: int = 1280,
                        overlap: int = 160, load_images: bool = True, seed: int = 0,
                        settings: BrowserSettings | None = None,
                        min_visible: float | Mapping[str, float] | None = None) -> DatasetSummary:
    """Render ``sources`` (``.html`` / ``.html.gz``) and write a YOLO dataset to ``out_dir``.

    Layout: ``images/{train,val}``, ``labels/{train,val}``, ``data.yaml`` and
    ``ground_truth.jsonl`` (one row per tile: image path, split, source page,
    visible regions with texts in tile pixel coordinates). Tiles without any
    card (page header / footer) are kept as background images.
    ``tile_height`` and ``overlap`` are CSS pixels.
    """
    from ..capture import capture_tiles, card_regions, render_html

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("images", "labels"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    train, val = split_sources(list(sources), val_sources=val_sources, val_fraction=val_fraction, seed=seed)
    if not train:
        raise ValueError("no training pages left after the split")
    summary = DatasetSummary(
        out_dir=out, data_yaml=out / "data.yaml", ground_truth=out / "ground_truth.jsonl",
        train_pages=[page_stem(p) for p in train], val_pages=[page_stem(p) for p in val],
    )
    counts: dict[str, Counter] = {"train": Counter(), "val": Counter()}
    settings = settings or BrowserSettings(headless=True)

    with summary.ground_truth.open("w", encoding="utf-8") as gt:
        for split, pages in (("train", train), ("val", val)):
            for src in pages:
                html = read_html(src)
                stem = page_stem(src)
                for width in viewport_widths:
                    for dsf in device_scale_factors:
                        tag = f"{stem}__w{width}_s{dsf:g}".replace(".", "p")
                        async with render_html(html, settings=settings, viewport_width=width,
                                               device_scale_factor=dsf, load_images=load_images) as page:
                            regions = await card_regions(page)
                            tiles = await capture_tiles(page, out / "images" / split, stem=tag,
                                                        tile_height=tile_height, overlap=overlap)
                        summary.renders += 1
                        if not regions:
                            log.warning("%s at width %d: no result cards found", stem, width)
                        for img_path, tile_box in tiles:
                            img_w, img_h = _png_size(img_path)
                            lines = regions_to_yolo(regions, tile_box, img_w, img_h, scale=dsf,
                                                    min_visible=min_visible)
                            (out / "labels" / split / f"{img_path.stem}.txt").write_text(
                                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                            for line in lines:
                                counts[split][VISION_CLASSES[int(line.split(" ", 1)[0])]] += 1
                            row = {
                                "image": str(img_path.relative_to(out)),
                                "split": split,
                                "source": stem,
                                "viewport_width": width,
                                "device_scale_factor": dsf,
                                "tile_box": tile_box.model_dump(),
                                "image_size": [img_w, img_h],
                                "regions": tile_ground_truth(regions, tile_box, scale=dsf, min_visible=min_visible),
                            }
                            gt.write(json.dumps(row) + "\n")
                            if split == "train":
                                summary.train_images += 1
                            else:
                                summary.val_images += 1
                        log.info("%s [%s] width=%d dsf=%g: %d cards, %d tiles",
                                 stem, split, width, dsf, len(regions), len(tiles))

    write_data_yaml(out, has_val=summary.val_images > 0)
    if summary.val_images == 0:
        log.warning("no validation pages: data.yaml validates on the training images (scores will be optimistic)")
    summary.instances = {split: {name: counts[split][name] for name in VISION_CLASSES} for split in counts}
    (out / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    return summary
