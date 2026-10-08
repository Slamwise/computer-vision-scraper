# The vision fallback: YOLO11 + OCR that labels its own training data

The DOM parser is exact, so it does almost all of the work. The vision path is
for two situations:

- **eBay changes its markup.** The parser then finds no cards on a page that
  is plainly a results page. The pipeline re-renders the cached HTML offline
  (no new request to eBay), screenshots it, and reads the screenshots.
- **All you have is a screenshot**, from a phone, a friend, or an old archive:
  `ebay-sold vision extract shot.png`.

## How labels come for free

The 2021 model needed boxes drawn by hand around prices and dates. This one
learns from pages the scraper has already saved:

1. `capture.render_html` renders saved HTML in Chromium with scripts off.
   eBay's stylesheets and photos come from its public CDNs and are cached on
   disk.
2. `capture.card_regions` asks the DOM where each result card, and each title,
   price, sold date, shipping line, condition and photo inside it, is drawn.
   It uses the parser's own rules: seller taglines aren't conditions, and a
   struck-through asking price isn't the sale price.
3. `capture.capture_tiles` screenshots the page as overlapping tiles.
   `vision.dataset` turns the boxes into YOLO labels for each tile and writes
   the matching ground-truth text for evaluation.

The labels are exactly what the browser drew, so every page in
`data/html-cache/` is free training data. A page the DOM parser reads correctly
today is also a ground-truth page for the vision model.

## Train your own model

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # Linux without GPU
pip install -e '.[vision]'

# 1. Dataset: render pages at several window widths and scale factors.
#    Hold out at least one page from a category you don't train on.
ebay-sold vision build-dataset data/html-cache/*.html \
    --val data/html-cache/<one page>.html \
    --out datasets/ebay --widths 1024,1280,1440,1920 --scales 1,2

# 2. Train. --install copies the best weights to data/models/ebay-sold-yolo.pt,
#    where scrape --vision fallback and vision extract look for them.
ebay-sold vision train datasets/ebay/data.yaml --epochs 30 --install

# 3. Score it against the DOM on pages it hasn't seen.
ebay-sold vision eval tests/fixtures/ebay/sold_2026-04-24_sears-roebuck-magazine.html.gz
```

On 4 CPU cores, one epoch over about 200 tiles takes about 2.5 minutes at
`imgsz 1024`. A GPU (`--device 0`) or a free Colab session cuts that to
seconds. The tests' archived pages (`tests/fixtures/ebay/*.html.gz`) are
enough to train a working model: four pages trained in 49 minutes found every
card on a held-out page.

Training settings that matter (already the defaults):

- **No flips, rotation, shear or mixup.** Mirrored text is not text.
- **`nbs` equals the batch size.** Ultralytics' default (64) accumulates 16
  small CPU batches per optimiser step, which leaves only a few dozen steps.
- **Mosaic is turned off for the last third of training**, so the final epochs
  see whole, real tiles.

## OCR

`get_ocr("auto")` uses **Tesseract** when the `tesseract` binary is installed,
and otherwise **RapidOCR** (pip-only, via onnxruntime).

- **Prices, dates, shipping and condition:** both engines read every one of
  them correctly on the held-out page.
- **Titles:** Tesseract is more exact. RapidOCR sometimes reads "&" as "8".

RapidOCR's own text detector garbles small crops: it drops spaces
("SoldApr25,2026") and warps letters. The engine here skips that detector. It
splits lines by ink, trims them, and feeds them to the recogniser at its native
48 px height. That took field accuracy from 61% to 100% and runs 12× faster.

Text read from pixels goes through `normalize` with `ocr=True`:

- **Lost decimal point.** eBay always prints cents, so "$3718" means $37.18.
- **Unsafe readings.** "$175.110" becomes unknown rather than $175,110.
- **Misread ones.** A "]" or "|" next to a digit in a date is read as a 1.

## Keeping the fallback useful

- **Keep the HTML cache.** When eBay changes its layout and the DOM parser
  starts finding nothing, pages saved before the change are still valid
  training data, and the vision fallback covers you while the parser is
  updated.
- **Add a few pages of a new layout and retrain.** The model trained only on
  the 2025–26 layout found every card on 2024 pages. Most of its errors there
  came from rows it had never seen, such as "Free returns" and "Best offer
  accepted".
- **Train on what you will feed it.** Add the widths and scale factors (`--scales 1,2`)
  of the screenshots you expect. Accuracy falls off outside the trained range.
- **Treat `confidence` as a review flag.** Listings from vision carry
  `extraction="vision"` and a confidence (detector × OCR). The database never
  lets a vision reading overwrite a DOM reading of the same sale.

## Licensing

Ultralytics (YOLO11) is AGPL-3.0, and so are weights trained from its
pretrained checkpoints. That's fine for personal use. Check the license before
you distribute weights or run this as a network service.
