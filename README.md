# computer-vision-scraper-yolov4

An end-to-end data extraction pipeline that scrapes eBay sold listings using YOLOv4-tiny object detection and Tesseract OCR -- no HTML parsing required.

## The Problem

eBay sold listings contain valuable market data (what sold, when, and for how much), but the page structure is fragile and changes frequently. Traditional CSS-selector scraping breaks every time eBay updates their layout. This project takes a different approach: treat the webpage as an image and use computer vision to find the data.

## How It Works

The pipeline has four stages:

### 1. Crawl & Screenshot

A headless Puppeteer browser navigates to eBay sold listing pages and captures full-page screenshots. The browser runs through rotating proxies with the `puppeteer-extra-plugin-stealth` package and randomized User-Agent strings to avoid bot detection.

```
node crawl.js [URL] [imageID] [proxyAddr] [proxyPort]
```

A second renderer (`renderhtml.js`) fetches raw HTML via API and renders it locally for cases where direct navigation is blocked.

### 2. Preprocess

Screenshots are batch-resized to a standardized 800px width using OpenCV (`resize.py`), normalizing input dimensions for consistent detection performance across different page layouts and screen captures.

### 3. Detect with YOLOv4-tiny

A custom-trained YOLOv4-tiny model runs darknet inference on every screenshot. The model was trained to draw bounding boxes around four listing elements:

| Class | What It Detects |
|-------|----------------|
| **price** | The sold price of the item |
| **date** | The date the item sold |
| **title** | The listing title / item name |
| **image** | The product thumbnail |

Detection runs at a 0.9 confidence threshold to minimize false positives. Results are written to `result.json` with bounding box coordinates in relative format.

```bash
# Run via tiny2.py, which configures darknet paths and executes:
./darknet detector test data/obj.data cfg/yolov4-tiny-obj.cfg weights/yolov4-tiny-obj_final.weights \
    -thresh 0.9 -ext_output -dont_show -out result.json < pathnames.txt
```

### 4. Crop & OCR

Bounding box coordinates from the detection JSON are converted from relative to absolute pixel positions, and each detected region is cropped from the original screenshot using OpenCV (`crop2.py`). Cropped images are sorted into category directories (`dates/`, `prices/`).

Finally, Tesseract OCR extracts text from each cropped bounding box:

```python
config = "-l eng --oem 1 --psm 7"  # Single line of text mode
text = pytesseract.image_to_string(Image.open(crop_path), config=config)
```

The result: structured data (price, date, title) extracted from a visual screenshot without ever parsing a single HTML element.

## Tech Stack

- **Scraping:** Puppeteer + puppeteer-extra (stealth plugin, proxy plugin, UA override)
- **Object Detection:** YOLOv4-tiny via darknet (custom-trained weights)
- **Image Processing:** OpenCV (Python) for resize, crop, and coordinate conversion
- **OCR:** Tesseract via pytesseract (English, LSTM engine, single-line mode)
- **Languages:** Python + Node.js

## Architecture

```
app/
  crawl.js       -- Puppeteer scraper with stealth + proxy rotation
  renderhtml.js  -- Alternative HTML renderer via API fetch
  resize.py      -- Batch image normalization (800px width)
  tiny2.py       -- Darknet YOLOv4-tiny inference runner
  crop2.py       -- Bounding box cropper (JSON → sorted image crops)
  tesstest.py    -- Tesseract OCR text extraction
  proxy.py       -- Proxy management utilities
```

## Status

Archived -- proof of concept. The pipeline works end-to-end on captured screenshots, successfully detecting and extracting price, date, and title data from eBay sold listings. However, eBay's CAPTCHA requirements during automated browsing prevented scaling to production-level data collection. The YOLOv4-tiny model and the vision-first extraction approach were validated; the bottleneck was access, not detection.

## License

MIT
