# ebay-sold

Collect eBay **sold-listing** prices into a SQLite database, then ask it
questions: what did this actually sell for, how spread out are the prices,
and how has that changed over time?

```text
$ ebay-sold import-html ~/Downloads/"Hot Wheels R34 Zamac for sale _ eBay.html"
Hot Wheels R34 Nissan Skyline GT-R ZAMAC: 70 listings from 1 page(s), 70 new

$ ebay-sold stats "hot wheels r34 nissan skyline gt-r zamac"
hot wheels r34 nissan skyline gt-r zamac (USD, price): 29 sold, 2 outliers trimmed
  median 49.99   middle 50% 41.99 - 56.78   mean 48.98
  min 21.50   max 75.00   sold 2026-01-31 .. 2026-04-25

$ ebay-sold stats "hot wheels r34 nissan skyline gt-r zamac" --by month
period     cur   sold     median        p25        p75        min        max
2026-01    USD      2      63.99      45.99      81.99      27.99      99.99
2026-02    USD      9      48.58      40.00      55.00      29.26      64.95
2026-03    USD      8      45.00      33.12      47.25      21.50      56.00
2026-04    USD     10      55.89      51.25      65.00      41.99      68.50

$ ebay-sold list "hot wheels r34 nissan skyline gt-r zamac" --limit 3
2026-04-25  USD 65.00 +9.45 ship [Brand New]  2012 2013 Hot Wheels Zamac Nissan Skyline R34 H/T 2000GT - X lot of 2 Zamacs
2026-04-24  USD 65.00 +free ship [Brand New]  Hot Wheels HW Showroom ZAMAC Nissan Skyline GT-R (R34) W/ Hot Wheels Protector
2026-04-24  USD 41.99 +5.83 ship [Brand New]  Hot Wheels 1:64 Diecast Nissan Skyline GT-R R34 2013 HW Showroom Zamac
```

(Real output on an archived eBay page from April 2026. Of the 70 cards on it,
eBay listed 39 under "Results matching fewer words", so stats count only the 31
exact matches.)

This is the second version of a 2021 project that read eBay screenshots with a
custom YOLOv4-tiny model and Tesseract (kept in [`legacy/`](legacy/)). That
version stopped working for three reasons: the trained weights were lost, eBay
kept asking it for CAPTCHAs, and its screenshots were never consistent.
[What changed](#what-changed-since-2021) explains how each one is fixed.

## How it works

```text
search ──▶ polite browser ──▶ HTML cache ──▶ DOM parser ──────────────▶ SQLite ──▶ stats / CSV / JSON
           (persistent profile,                │  finds nothing?
            human pacing,                      ▼  (eBay changed its markup)
            stops when challenged)        render cached HTML offline ──▶ YOLO11 + OCR ──┘
                                               │  still nothing? (optional)
                                               └──────────────────▶ Claude vision ───┘
```

1. **Fetching** uses one real Chromium/Chrome with a persistent profile. It
   paces itself like a person, loads 240 results per page, never fetches the
   same page twice, and stops instead of retrying when eBay shows a challenge.
   See [docs/anti-bot.md](docs/anti-bot.md).
2. **The DOM parser** reads every field the result card shows: price, price
   range, struck-through list price, shipping, sold date, condition,
   auction/Buy It Now/Best Offer, bids, seller and feedback, and location. It
   also notes whether a card sits under eBay's "Results matching fewer words"
   divider. It handles eBay's current layout and the older one.
3. **The vision fallback** takes over when the parser finds nothing on a page
   that isn't a challenge, which usually means eBay changed its markup. It
   re-renders the *cached* HTML offline (no extra request to eBay),
   screenshots it, and reads the screenshots with a YOLO11 detector and OCR.
   The detector trains itself from saved pages, as
   [the next section](#the-vision-fallback-trains-itself) explains.
4. **Claude** (optional, needs an API key) can read the same screenshots
   when vision also comes up empty.

## Quick start

Requires Python 3.10+.

```bash
git clone <this repo> && cd <this repo>
python -m venv .venv && source .venv/bin/activate

pip install -e .                    # scraping, parsing, database
playwright install chromium         # or use your installed Chrome with --chrome

# Optional: the vision fallback. On Linux without a GPU, install CPU torch first (much smaller):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[vision]'
# Optional, for better titles from OCR: brew install tesseract / sudo apt install tesseract-ocr

# Optional: Claude screenshot reader
pip install -e '.[llm]' && export ANTHROPIC_API_KEY=...
```

Then:

```bash
ebay-sold browser                         # once: open the scraper's profile, browse a bit, accept cookies
ebay-sold scrape "giannis prizm 194" --pages 2 --chrome
ebay-sold stats "giannis prizm 194" --since 30
ebay-sold list "giannis prizm 194" --limit 20
ebay-sold export --format csv --out sold.csv
```

Run it from your own computer and home connection. eBay blocks cloud and
datacenter IPs at the edge no matter what the browser does.

### Zero-risk mode: import pages you saved yourself

Open the sold search in your normal browser, save the page (Ctrl+S / Cmd+S),
and import it:

```bash
ebay-sold import-html ~/Downloads/*.html          # keywords are read from each page
```

## Commands

| command | what it does |
|---|---|
| `scrape KEYWORDS... [--pages N]` | Search sold listings and store them. Several searches share one browser session. Filters: `--site www.ebay.co.uk`, `--category`, `--condition used`, `--min-price/--max-price`, `--exclude psa lot`, `--auction` / `--buy-it-now`. |
| `import-html FILES...` | Parse result pages saved from your browser (`.html` or `.html.gz`). |
| `stats [KEYWORDS]` | Median, middle 50%, mean, min/max and date range, per currency. `--by month` (or `week`, `day`) shows the trend. Outliers beyond 1.5×IQR are trimmed (`--no-trim` keeps them). `--include-shipping` uses price + shipping. `--json` for scripts. |
| `list [KEYWORDS]` / `export --format csv\|json --out FILE` | Read the stored listings. Filters: `--since 30` (days) or a date, `--until`, `--condition "Pre-Owned"`, `--currency`, `--all-matches`. |
| `browser` | Open the scraper's own browser profile, to warm it up or check on a challenge. |
| `vision build-dataset / train / extract / eval` | Train and use the YOLO + OCR fallback ([docs/vision.md](docs/vision.md)). |
| `llm extract IMAGES...` | Read listings from screenshots with Claude. |

By default, stats and exports include only **exact matches**. eBay pads thin
searches with "Results matching fewer words", which are often different items.
`--all-matches` includes them anyway.

## What changed since 2021

| 2021 problem | Cause | Now |
|---|---|---|
| CAPTCHAs and bot flags | A random **Firefox** user agent on **Chromium** (an instant fingerprint mismatch), free proxy-list IPs, a new cookie-less browser per page, 60 results per page, no pacing | One persistent profile with the browser's real user agent, 240 results per page, an HTML cache, jittered pacing with page budgets, and challenge detection that stops the run and waits for you to solve it in the window. Details: [docs/anti-bot.md](docs/anti-bot.md). |
| Inconsistent screenshots | No fixed viewport, an arbitrary 5 s wait, lazy images and fonts half loaded, sticky headers stamped across the page, and `resize.py` squashing every image to the shortest one's height | Fixed viewport and scale, fonts and images awaited, animations off, fixed and sticky overlays hidden, and the page captured as overlapping tiles. Each tile is bigger than a card, so every card appears whole in at least one. Nothing is resized by hand. |
| No working YOLO/OCR pipeline | Darknet build, hand-labelled data and weights were lost | Ultralytics YOLO11 (`pip install`), trained on labels generated automatically from saved pages, plus OCR that reads every price and date correctly on held-out pages |
| Fragile selectors | (the reason vision was chosen) | The DOM is the primary path because it is exact. Vision is the safety net, and it was tested on a layout it never saw (below). |

## The vision fallback trains itself

The 2021 model needed hand-drawn boxes. This one doesn't. When a saved results
page is rendered, the browser already knows where every card, title, price,
sold date, shipping line, condition and photo is drawn. `vision build-dataset`
turns those positions into YOLO labels, so every page the scraper has ever
cached is free training data.

```bash
ebay-sold vision build-dataset data/html-cache/*.html --out datasets/ebay --scales 1,2
ebay-sold vision train datasets/ebay/data.yaml --install     # about 50 min on a 4-core CPU; minutes on a GPU
ebay-sold vision extract screenshot.png                     # read any screenshot
```

Measured on 4 CPU cores, with 4 archived pages for training, 15 epochs and about 49 minutes:

| test | cards found | price | sold date | shipping | condition | title |
|---|---|---|---|---|---|---|
| held-out page, same layout (64 cards), widths 1024–1920 | 100% | 100% | 100% | 100% | 100% | 0.99 mean similarity |
| **eBay's older 2024 layout, never trained on** (157 cards) | 100% | about 90% of sales had price, date and shipping all right | | | | 97–99% of titles ≥ 0.9 similar |

On the older layout, most errors were a neighbouring row ("or Best Offer") read
as shipping, or OCR dropping a decimal point. The OCR repairs now catch the
second kind. Retraining on a few pages of a new layout removes the first.
Details, failure cases and the training recipe are in
[docs/vision.md](docs/vision.md).

## The Claude screenshot reader (optional)

`--llm fallback` on `scrape`, or `ebay-sold llm extract shot.png`, sends
screenshots to Claude with a strict JSON schema. Claude copies the visible text
verbatim, and the same normalizer as the other paths turns it into values. The
default model is `claude-opus-5-5` at low effort. That costs roughly **$0.80 for a
70-result page and $2.50 for a 240-result page**, so it's a last resort. Use
`--model` to choose another model. Screenshots showing a CAPTCHA stop the run,
just like a detected challenge.

## Data

Everything lives under `data/` (change it with `--data-dir` or `EBAY_SOLD_DATA_DIR`):

| path | contents |
|---|---|
| `ebay_sold.sqlite` | listings (one row per item, keyed by site + item id), searches, and observations (which search saw which item, on what page and position, as an exact match or not) |
| `html-cache/` | every fetched results page, so you can re-parse or train on it later |
| `browser-profile/` | the scraper's cookies and history |
| `blocked/` | challenge pages that stopped a run, and the cooldown that makes the next run wait |
| `models/ebay-sold-yolo.pt` | the vision model, once trained |

When the same sale is read several ways, DOM data wins over Claude, and Claude
wins over YOLO+OCR. A missing value never erases a known one.

## Configuration

Optional `ebay-sold.toml` in the working directory:

```toml
data_dir = "data"

[browser]
channel = "chrome"     # drive your installed Google Chrome
headless = false       # a visible window gets fewer challenges and lets you solve one

[pacing]
min_delay_s = 6
max_delay_s = 18
max_pages_per_run = 25
```

Every setting and its default is in [`src/ebay_sold/config.py`](src/ebay_sold/config.py).

## Development

```bash
pip install -e '.[vision,llm,dev]'
pytest                          # everything; tests needing a browser or models skip themselves if unavailable
pytest -m "not browser"         # fast unit tests only
```

The tests never contact eBay. The fixtures in `tests/fixtures/ebay/` are real
eBay results pages archived by the Internet Archive's Wayback Machine (2024
legacy layout and 2025–26 current layout), with scripts stripped. Their sources
are listed in `manifest.json`. End-to-end scraping tests run the real browser
against a local server that imitates eBay, including challenge pages, errors
and redirects.

## Terms of use

Scraping eBay is against eBay's User Agreement. Keep it to personal research
volumes, don't sign in to your eBay account in the scraper's profile, stop if
eBay blocks you or asks you to, and don't republish the data.
[docs/anti-bot.md](docs/anti-bot.md#terms-of-use) explains why, and covers the
official API route.

## License

MIT. The optional vision extra uses [Ultralytics](https://github.com/ultralytics/ultralytics),
which is AGPL-3.0. That matters if you distribute trained weights or run this
as a service.
