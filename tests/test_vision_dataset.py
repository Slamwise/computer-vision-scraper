"""Auto-labelling: DOM regions -> per-tile YOLO labels, data.yaml, split by page."""

from __future__ import annotations

import json

import pytest

from ebay_sold.models import VISION_CLASSES, Box, CardRegion
from ebay_sold.vision.dataset import (
    build_dataset,
    clip_box,
    regions_to_yolo,
    split_sources,
    tile_ground_truth,
    write_data_yaml,
)

from conftest import EBAY_FIXTURES


def _parse(lines: list[str]) -> list[tuple[str, float, float, float, float]]:
    out = []
    for line in lines:
        c, cx, cy, w, h = line.split()
        out.append((VISION_CLASSES[int(c)], float(cx), float(cy), float(w), float(h)))
    return out


def _card(y: float, *, h: float = 200.0, item_id: str = "1") -> CardRegion:
    return CardRegion(
        item_id=item_id,
        box=Box(x=100, y=y, w=800, h=h),
        fields={
            "sold_date": Box(x=400, y=y + 10, w=120, h=16),
            "title": Box(x=400, y=y + 30, w=400, h=20),
            "price": Box(x=400, y=y + 80, w=80, h=24),
            "image": Box(x=110, y=y + 10, w=180, h=180),
        },
        texts={"sold_date": "Sold Apr 25, 2026", "title": "A thing", "price": "$65.00"},
    )


def test_clip_box():
    tile = Box(x=0, y=1000, w=1280, h=1280)
    assert clip_box(Box(x=10, y=900, w=100, h=200), tile) == Box(x=10, y=1000, w=100, h=100)
    assert clip_box(Box(x=10, y=10, w=100, h=100), tile) is None


def test_regions_to_yolo_fully_inside_tile():
    tile = Box(x=0, y=1000, w=1280, h=1280)
    lines = regions_to_yolo([_card(1100)], tile, 1280, 1280)
    parsed = _parse(lines)
    assert [p[0] for p in parsed] == ["listing", "title", "price", "sold_date", "image"]  # VISION_CLASSES order
    cls, cx, cy, w, h = parsed[0]
    assert cx == pytest.approx((100 + 400) / 1280)
    assert cy == pytest.approx((100 + 100) / 1280)  # card top is 100 px into the tile
    assert w == pytest.approx(800 / 1280) and h == pytest.approx(200 / 1280)
    assert all(0.0 <= v <= 1.0 for p in parsed for v in p[1:])


def test_clipping_and_visibility_thresholds():
    tile = Box(x=0, y=0, w=1280, h=1280)
    # Card starts 120 px above the tile: 80/200 = 40 % of it is visible (>= 30 %, kept, clipped);
    # sold_date (y -110..-94) is outside, title (y -90..-70) outside, price (y -40..-16) outside,
    # image (y -110..70): 70/180 = 39 % visible (>= 30 %, kept).
    lines = regions_to_yolo([_card(-120)], tile, 1280, 1280)
    parsed = {p[0]: p for p in _parse(lines)}
    assert set(parsed) == {"listing", "image"}
    _, cx, cy, w, h = parsed["listing"]
    assert cy - h / 2 == pytest.approx(0.0, abs=1e-6)  # clipped at the tile's top edge
    assert h == pytest.approx(80 / 1280)

    # A text field cut in half: 9/16 visible -> kept; 7/16 visible -> dropped.
    card = _card(500)
    card.fields = {"sold_date": Box(x=400, y=1271, w=120, h=16)}
    assert [p[0] for p in _parse(regions_to_yolo([card], tile, 1280, 1280))] == ["listing", "sold_date"]
    card.fields = {"sold_date": Box(x=400, y=1273, w=120, h=16)}
    assert [p[0] for p in _parse(regions_to_yolo([card], tile, 1280, 1280))] == ["listing"]

    # Card only 20 % visible: its box is dropped, but its fields are judged on their own
    # (sold_date is whole, title exactly half visible, the photo only 17 %).
    names = [p[0] for p in _parse(regions_to_yolo([_card(1240)], tile, 1280, 1280))]
    assert names == ["title", "sold_date"]
    # min_visible override
    assert _parse(regions_to_yolo([_card(1240)], tile, 1280, 1280, min_visible=0.1))[0][0] == "listing"


def test_device_scale_factor_keeps_normalised_coordinates():
    tile = Box(x=0, y=640, w=1280, h=1280)
    one = _parse(regions_to_yolo([_card(700)], tile, 1280, 1280, scale=1.0))
    two = _parse(regions_to_yolo([_card(700)], tile, 2560, 2560, scale=2.0))
    assert len(one) == len(two)
    for a, b in zip(one, two):
        assert a[0] == b[0]
        assert a[1:] == pytest.approx(b[1:])
    # A wrong scale would put boxes in the wrong place: half-size boxes at dsf 2 with scale=1.
    wrong = _parse(regions_to_yolo([_card(700)], tile, 2560, 2560, scale=1.0))
    assert wrong[0][3] == pytest.approx(one[0][3] / 2)


def test_tile_ground_truth_is_in_tile_pixels():
    tile = Box(x=0, y=1000, w=1280, h=1280)
    gt = tile_ground_truth([_card(1100)], tile, scale=2.0)
    assert len(gt) == 1
    assert gt[0]["box"] == {"x": 200.0, "y": 200.0, "w": 1600.0, "h": 400.0}
    assert gt[0]["texts"]["price"] == "$65.00"
    assert gt[0]["fields"]["price"]["visible"] == 1.0


def test_data_yaml_lists_classes_in_order(tmp_path):
    path = write_data_yaml(tmp_path)
    text = path.read_text()
    assert f"path: {tmp_path.resolve()}" in text
    assert "train: images/train" in text and "val: images/val" in text
    names = [line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("  ")]
    assert tuple(names) == VISION_CLASSES
    assert "val: images/train" in write_data_yaml(tmp_path, has_val=False).read_text()


def test_split_is_by_page(tmp_path):
    pages = [tmp_path / f"p{i}.html.gz" for i in range(10)]
    train, val = split_sources(pages, val_fraction=0.2, seed=1)
    assert len(val) == 2 and len(train) == 8
    assert not set(train) & set(val) and set(train) | set(val) == set(pages)
    assert split_sources(pages, val_fraction=0.2, seed=1) == (train, val)  # deterministic
    train, val = split_sources(pages, val_sources=[pages[3]])
    assert val == [pages[3]] and pages[3] not in train and len(train) == 9
    # A single page is never held out (there would be nothing to train on).
    assert split_sources(pages[:1], val_fraction=0.5) == (pages[:1], [])


@pytest.mark.browser
@pytest.mark.vision
async def test_build_dataset_from_fixture(require_browser, require_vision, tmp_path):
    train_page = EBAY_FIXTURES / "sold_2026-04-26_hot-wheels-r34-zamac.html.gz"
    val_page = EBAY_FIXTURES / "sold_2026-04-16_ta1-adapter.html.gz"
    out = tmp_path / "ds"
    summary = await build_dataset([train_page, val_page], out, val_sources=[val_page], viewport_widths=(1280,),
                                  load_images=False, tile_height=1280, overlap=160)
    assert summary.train_pages == ["sold_2026-04-26_hot-wheels-r34-zamac"]
    assert summary.val_pages == ["sold_2026-04-16_ta1-adapter"]
    assert summary.train_images > 3 and summary.val_images > 1
    # 70 visible cards on the Hot Wheels page; cards cut by tile edges are counted in both tiles.
    assert summary.instances["train"]["listing"] >= 70
    assert summary.instances["train"]["price"] >= 70

    for split in ("train", "val"):
        images = sorted((out / "images" / split).glob("*.png"))
        labels = sorted((out / "labels" / split).glob("*.txt"))
        assert [p.stem for p in images] == [p.stem for p in labels]
        for lbl in labels:
            for line in lbl.read_text().splitlines():
                c, *vals = line.split()
                assert 0 <= int(c) < len(VISION_CLASSES)
                assert all(0.0 <= float(v) <= 1.0 for v in vals)

    rows = [json.loads(line) for line in (out / "ground_truth.jsonl").read_text().splitlines()]
    assert len(rows) == summary.train_images + summary.val_images
    # Split by page: every tile of a page lands in the same split.
    splits_by_page: dict[str, set[str]] = {}
    for r in rows:
        splits_by_page.setdefault(r["source"], set()).add(r["split"])
        assert (out / r["image"]).exists()
    assert splits_by_page == {"sold_2026-04-26_hot-wheels-r34-zamac": {"train"}, "sold_2026-04-16_ta1-adapter": {"val"}}
    first = next(reg for r in rows for reg in r["regions"] if reg["item_id"] == "358473128518")
    assert first["texts"]["price"] == "$65.00"
    assert "names:" in (out / "data.yaml").read_text()
