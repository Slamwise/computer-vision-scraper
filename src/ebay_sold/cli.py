"""``ebay-sold`` command line.

    ebay-sold scrape "hot wheels r34 zamac" --pages 2
    ebay-sold import-html ~/Downloads/*.html --keywords "hot wheels r34 zamac"
    ebay-sold stats "hot wheels r34 zamac" --since 30
    ebay-sold export --format csv --out sold.csv
    ebay-sold browser
    ebay-sold vision build-dataset tests/fixtures/ebay/*.html.gz --out datasets/ebay
    ebay-sold vision train datasets/ebay/data.yaml --epochs 40
    ebay-sold vision extract screenshot.png
    ebay-sold llm extract screenshot.png
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import Settings, load_settings
from .models import Listing, SearchQuery


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s" if args.verbose else "%(message)s",
    )
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ebay-sold", description="Collect eBay sold-listing prices into SQLite.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--config", help="path to ebay-sold.toml (default: ./ebay-sold.toml if present)")
    p.add_argument("--data-dir", help="where the database, browser profile and caches live (default: ./data)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(title="commands", metavar="COMMAND")

    s = sub.add_parser("scrape", help="search sold listings and store them")
    s.add_argument("keywords", nargs="+", help="one or more searches, e.g. \"hot wheels r34 zamac\"")
    s.add_argument("--pages", type=int, default=1, help="max result pages per search (240 results each)")
    _add_query_filters(s)
    s.add_argument("--headless", action="store_true", help="no browser window (more challenges; none can be solved by hand)")
    s.add_argument("--chrome", action="store_true", help="drive your installed Google Chrome")
    s.add_argument("--proxy", help="proxy URL, e.g. http://user:pass@host:port (see docs/anti-bot.md)")
    s.add_argument("--no-cache", action="store_true", help="refetch even if the page is in the HTML cache")
    s.add_argument("--screenshots", action="store_true", help="save page screenshots (tiles) next to the data")
    s.add_argument("--vision", choices=["off", "fallback", "always"], default="fallback",
                   help="YOLO+OCR on screenshots: only when the DOM parser finds nothing (default), never, or always")
    s.add_argument("--llm", choices=["off", "fallback", "always"], default="off",
                   help="ask Claude to read screenshots when DOM and vision both fail (needs ANTHROPIC_API_KEY)")
    s.set_defaults(func=cmd_scrape)

    s = sub.add_parser("import-html", help="parse result pages you saved from your own browser")
    s.add_argument("files", nargs="+", type=Path)
    s.add_argument("--keywords", help="search keywords (default: read from each page)")
    s.add_argument("--site", default="www.ebay.com")
    s.set_defaults(func=cmd_import_html)

    s = sub.add_parser("browser", help="open the scraper's browser profile to warm it up or check a challenge")
    s.add_argument("--url", default="https://www.ebay.com/")
    s.add_argument("--chrome", action="store_true")
    s.set_defaults(func=cmd_browser)

    s = sub.add_parser("stats", help="sold-price statistics for a search")
    s.add_argument("keywords", nargs="?", help="keywords of a previous scrape (default: everything)")
    _add_listing_filters(s)
    s.add_argument("--include-shipping", action="store_true", help="price + shipping")
    s.add_argument("--no-trim", action="store_true", help="keep outliers (default trims beyond 1.5x IQR)")
    s.add_argument("--by", choices=["day", "week", "month"], help="one line per period, to see the trend")
    s.add_argument("--json", action="store_true", help="machine-readable output")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("list", help="print stored listings")
    s.add_argument("keywords", nargs="?")
    _add_listing_filters(s)
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("export", help="export stored listings to CSV or JSON")
    s.add_argument("keywords", nargs="?")
    _add_listing_filters(s)
    s.add_argument("--format", choices=["csv", "json"], default="csv")
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(func=cmd_export)

    v = sub.add_parser("vision", help="YOLO + OCR fallback: datasets, training, extraction")
    vsub = v.add_subparsers(title="vision commands", metavar="COMMAND")

    s = vsub.add_parser("build-dataset", help="render saved result pages and auto-label them for YOLO")
    s.add_argument("html", nargs="+", type=Path, help=".html or .html.gz result pages (data/html-cache works)")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--val", nargs="*", type=Path, default=None, help="pages to hold out for validation")
    s.add_argument("--widths", default="1024,1280,1440", help="viewport widths to render, comma-separated")
    s.add_argument("--scales", default="1", help="device scale factors, e.g. 1,2 to also cover HiDPI screenshots")
    s.add_argument("--no-images", action="store_true", help="gray boxes instead of product photos (offline)")
    s.set_defaults(func=cmd_vision_build)

    s = vsub.add_parser("train", help="train the detector on a built dataset")
    s.add_argument("data_yaml", type=Path)
    s.add_argument("--epochs", type=int, default=30)
    s.add_argument("--imgsz", type=int, default=1024)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--model", default="yolo11n.pt", help="starting weights")
    s.add_argument("--device", default="cpu", help="cpu, 0 (first GPU), mps")
    s.add_argument("--install", action="store_true", help="copy best weights to the default model path")
    s.set_defaults(func=cmd_vision_train)

    s = vsub.add_parser("extract", help="read sold listings from screenshots")
    s.add_argument("images", nargs="+", type=Path)
    s.add_argument("--weights", type=Path)
    s.add_argument("--site", default="www.ebay.com")
    s.add_argument("--save", metavar="KEYWORDS", help="store the listings under this search")
    s.set_defaults(func=cmd_vision_extract)

    s = vsub.add_parser("eval", help="score the detector + OCR against the DOM on saved pages")
    s.add_argument("html", nargs="+", type=Path)
    s.add_argument("--weights", type=Path)
    s.add_argument("--width", type=int, default=1280)
    s.set_defaults(func=cmd_vision_eval)

    lp = sub.add_parser("llm", help="Claude vision extraction (optional, needs ANTHROPIC_API_KEY)")
    lsub = lp.add_subparsers(title="llm commands", metavar="COMMAND")
    s = lsub.add_parser("extract", help="read sold listings from screenshots with Claude")
    s.add_argument("images", nargs="+", type=Path)
    s.add_argument("--site", default="www.ebay.com")
    s.add_argument("--model", help="Claude model id (default from settings)")
    s.add_argument("--save", metavar="KEYWORDS", help="store the listings under this search")
    s.set_defaults(func=cmd_llm_extract)
    return p


def _add_query_filters(s: argparse.ArgumentParser) -> None:
    s.add_argument("--site", default="www.ebay.com", help="www.ebay.com, www.ebay.co.uk, www.ebay.de, ...")
    s.add_argument("--category", type=int, help="eBay category id")
    s.add_argument("--condition", choices=["new", "used", "open_box", "refurbished", "for_parts"])
    s.add_argument("--min-price", type=float)
    s.add_argument("--max-price", type=float)
    s.add_argument("--exclude", nargs="*", default=[], help="words to exclude, e.g. --exclude psa lot")
    fmt = s.add_mutually_exclusive_group()
    fmt.add_argument("--auction", action="store_true", help="auctions only")
    fmt.add_argument("--buy-it-now", action="store_true", help="Buy It Now only")
    s.add_argument("--sort", choices=["ended_recently", "price_high", "price_low", "best_match"], default="ended_recently")


def _add_listing_filters(s: argparse.ArgumentParser) -> None:
    s.add_argument("--since", help="sold on/after: a date (2026-04-01) or a number of days (30)")
    s.add_argument("--until", help="sold on/before (date)")
    s.add_argument("--condition", help="exact condition text, e.g. \"Brand New\" or \"Pre-Owned\"")
    s.add_argument("--currency", help="ISO code, e.g. USD")
    s.add_argument("--site")
    s.add_argument("--all-matches", action="store_true",
                   help="include listings eBay showed under 'Results matching fewer words'")


# --- commands ----------------------------------------------------------------


def _settings(args: argparse.Namespace, **overrides: Any) -> Settings:
    if args.data_dir:
        overrides["data_dir"] = args.data_dir
    return load_settings(args.config, **overrides)


def _db(settings: Settings):
    from .db import Database

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return Database(settings.db_path)


def cmd_scrape(args: argparse.Namespace) -> int:
    from .pipeline import scrape

    overrides: dict[str, Any] = {}
    if args.headless:
        overrides["browser.headless"] = True
    if args.chrome:
        overrides["browser.channel"] = "chrome"
    if args.proxy:
        overrides["browser.proxy_server"] = args.proxy
    settings = _settings(args, **overrides)
    queries = [
        SearchQuery(
            keywords=kw,
            site=args.site,
            category_id=args.category,
            condition=args.condition,
            min_price=args.min_price,
            max_price=args.max_price,
            exclude=args.exclude,
            listing_type="auction" if args.auction else "buy_it_now" if args.buy_it_now else "all",
            sort=args.sort,
        )
        for kw in args.keywords
    ]

    def on_page(query: SearchQuery, page) -> None:
        source = "cache" if page.from_cache else "fetched"
        print(f"  [{query.keywords}] page {page.page}: {page.listings} listings "
              f"({page.exact_matches} exact matches, {page.new} new, via {page.extraction}, {source})")

    with _db(settings) as db:
        reports = asyncio.run(scrape(
            queries, settings=settings, db=db, pages=args.pages, use_cache=not args.no_cache,
            screenshots=args.screenshots, vision=args.vision, llm=args.llm, on_page=on_page,
        ))
    status = 0
    for r in reports:
        total = f", eBay reports {r.total_results} results" if r.total_results is not None else ""
        print(f"{r.keywords}: {r.listings} listings ({r.new} new){total}; {r.stopped_reason}")
        if r.block_kind:
            status = 3
            print("  eBay asked for verification. Wait a while before the next run; "
                  "see docs/anti-bot.md. `ebay-sold browser` opens the same profile so you can check by hand.")
    if len(reports) < len(queries):
        print(f"{len(queries) - len(reports)} searches not started.")
    return status


def cmd_import_html(args: argparse.Namespace) -> int:
    from .pipeline import import_html

    settings = _settings(args)
    missing = [f for f in args.files if not f.exists()]
    if missing:
        print("not found: " + ", ".join(map(str, missing)), file=sys.stderr)
        return 2
    with _db(settings) as db:
        reports = import_html(args.files, db=db, keywords=args.keywords, site=args.site)
    for r in reports:
        print(f"{r.keywords}: {r.listings} listings from {len(r.pages)} page(s), {r.new} new")
    return 0 if any(r.listings for r in reports) else 1


def cmd_browser(args: argparse.Namespace) -> int:
    from .profile import open_profile_browser

    overrides = {"browser.headless": False}
    if args.chrome:
        overrides["browser.channel"] = "chrome"
    settings = _settings(args, **overrides)
    settings.ensure_dirs()
    print(f"Opening the scraper's browser profile ({settings.profile_dir}). Close the window when you are done.")
    asyncio.run(open_profile_browser(settings, args.url))
    return 0


def _filters(args: argparse.Namespace) -> dict[str, Any]:
    since = None
    if args.since:
        since = date.today() - timedelta(days=int(args.since)) if args.since.isdigit() else date.fromisoformat(args.since)
    return {
        "keywords": getattr(args, "keywords", None),
        "since": since,
        "until": date.fromisoformat(args.until) if args.until else None,
        "condition": args.condition,
        "currency": args.currency,
        "site": args.site,
        "exact_matches_only": not args.all_matches,
    }


def cmd_stats(args: argparse.Namespace) -> int:
    settings = _settings(args)
    if args.by:
        return _stats_by_period(args, settings)
    with _db(settings) as db:
        stats = db.price_stats(**_filters(args), include_shipping=args.include_shipping, trim_outliers=not args.no_trim)
    if args.json:
        print(json.dumps({cur: s.model_dump(mode="json") for cur, s in stats.items()}, indent=2))
        return 0
    if not stats:
        print("no sold listings match")
        return 1
    label = "price + shipping" if args.include_shipping else "price"
    for cur, s in stats.items():
        trimmed = f", {s.outliers_removed} outliers trimmed" if s.outliers_removed else ""
        print(f"{args.keywords or 'all listings'} ({cur}, {label}): {s.count} sold{trimmed}")
        print(f"  median {_money(s.median)}   middle 50% {_money(s.p25)} - {_money(s.p75)}   "
              f"mean {_money(s.mean)}")
        print(f"  min {_money(s.min)}   max {_money(s.max)}   sold {s.first_sold} .. {s.last_sold}")
    return 0


def _stats_by_period(args: argparse.Namespace, settings: Settings) -> int:
    with _db(settings) as db:
        rows = db.price_series(period=args.by, include_shipping=args.include_shipping,
                               trim_outliers=not args.no_trim, **_filters(args))
    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0 if rows else 1
    if not rows:
        print("no dated sold listings match")
        return 1
    print(f"{'period':<10} {'cur':<4} {'sold':>5} {'median':>10} {'p25':>10} {'p75':>10} {'min':>10} {'max':>10}")
    for r in rows:
        print(f"{r['period']:<10} {r['currency']:<4} {r['count']:>5} {_money(r['median']):>10} "
              f"{_money(r['p25']):>10} {_money(r['p75']):>10} {_money(r['min']):>10} {_money(r['max']):>10}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    settings = _settings(args)
    with _db(settings) as db:
        rows = db.query_listings(**_filters(args), limit=args.limit)
    for item in rows:
        print(_listing_line(item))
    return 0 if rows else 1


def cmd_export(args: argparse.Namespace) -> int:
    settings = _settings(args)
    with _db(settings) as db:
        if args.format == "csv":
            n = db.export_csv(args.out, **_filters(args))
        else:
            n = db.export_json(args.out, **_filters(args))
    print(f"wrote {n} listings to {args.out}")
    return 0


def cmd_vision_build(args: argparse.Namespace) -> int:
    from .vision.dataset import build_dataset

    widths = tuple(int(w) for w in args.widths.split(",") if w.strip())
    scales = tuple(float(x) for x in args.scales.split(",") if x.strip())
    summary = asyncio.run(build_dataset(
        list(args.html), args.out, val_sources=args.val, viewport_widths=widths, device_scale_factors=scales,
        load_images=not args.no_images,
    ))
    print(summary.model_dump_json(indent=2))
    print(f"next: ebay-sold vision train {summary.data_yaml} --install")
    return 0


def cmd_vision_train(args: argparse.Namespace) -> int:
    import shutil

    from .vision.detector import train

    settings = _settings(args)
    # nbs = batch: one optimiser step per batch. Ultralytics' default (64) would
    # accumulate 16 small CPU batches per step and leave a few dozen steps in total.
    best = train(args.data_yaml, model=args.model, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                 nbs=args.batch, device=args.device, project=settings.data_dir / "runs")
    print(f"best weights: {best}")
    if args.install:
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, settings.weights_path)
        print(f"installed as {settings.weights_path}")
    return 0


def cmd_vision_extract(args: argparse.Namespace) -> int:
    from .vision.detector import Detector
    from .vision.extract import extract_listings
    from .vision.ocr import get_ocr

    settings = _settings(args)
    weights = args.weights or settings.weights_path
    if not Path(weights).exists():
        print(f"no model at {weights}; train one first (docs/vision.md)", file=sys.stderr)
        return 2
    detector = Detector(weights, imgsz=settings.vision.imgsz, conf=settings.vision.conf)
    ocr = get_ocr(settings.vision.ocr_backend)
    found: list[Listing] = []
    for image in args.images:
        listings = extract_listings(image, detector=detector, ocr=ocr, site=args.site)
        print(f"{image}: {len(listings)} listings")
        for item in listings:
            print("  " + _listing_line(item))
        found.extend(listings)
    return _maybe_save(args, settings, found)


def cmd_vision_eval(args: argparse.Namespace) -> int:
    from .vision.evaluate import evaluate

    settings = _settings(args)
    weights = args.weights or settings.weights_path
    report = evaluate(weights, list(args.html), viewport_width=args.width, ocr_backend=settings.vision.ocr_backend)
    print(report.summary())
    return 0


def cmd_llm_extract(args: argparse.Namespace) -> int:
    from .llm import ClaudeExtractor

    settings = _settings(args)
    llm_settings = settings.llm.model_copy(update={"model": args.model}) if args.model else settings.llm
    extractor = ClaudeExtractor(llm_settings)
    found: list[Listing] = []
    for image in args.images:
        listings = extractor.extract(image, site=args.site)
        print(f"{image}: {len(listings)} listings")
        for item in listings:
            print("  " + _listing_line(item))
        found.extend(listings)
    return _maybe_save(args, settings, found)


def _maybe_save(args: argparse.Namespace, settings: Settings, listings: list[Listing]) -> int:
    if not args.save:
        return 0 if listings else 1
    with _db(settings) as db:
        search_id = db.record_search(SearchQuery(keywords=args.save, site=args.site), "screenshot")
        stats = db.upsert_listings(listings, search_id=search_id)
    print(f"saved under {args.save!r}: {stats.new} new, {stats.updated} updated")
    return 0


def _money(value: float | None) -> str:
    return "-" if value is None else f"{value:,.2f}"


def _listing_line(item: Listing) -> str:
    price = _money(item.price) + (f"-{_money(item.price_max)}" if item.price_max else "")
    ship = "" if item.shipping is None else (" +free ship" if item.shipping == 0 else f" +{_money(item.shipping)} ship")
    when = item.sold_date.isoformat() if item.sold_date else "????-??-??"
    cond = f" [{item.condition}]" if item.condition else ""
    return f"{when}  {item.currency or ''} {price}{ship}{cond}  {item.title[:80]}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
