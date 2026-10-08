"""Read sold listings off screenshots with Claude (vision).

This is the extractor of last resort, and a cross-check for the other two. The
DOM parser is exact but breaks when eBay changes markup; the YOLO + OCR path
needs a trained model and misreads small text. Claude reads a screenshot the way
a person would, so it keeps working through layout changes and on screenshots
people took themselves. It costs money per page and is slower, so it is opt-in.

Design choices:

* **Transcribe, never interpret.** The model copies the text each card shows
  ("$65.00", "Sold  Apr 25, 2026", "+$9.45 delivery") into a strict JSON schema
  (structured outputs) and returns ``null`` for anything it cannot see. Typed
  values come only from ``ebay_sold.normalize``, the same code the DOM and OCR
  paths use, so all three extractors agree on what a string means and a model
  cannot invent a number.
* **Small, overlapping slices.** A results page is tens of thousands of pixels
  tall. Sent whole, it would be downscaled until the small text is unreadable
  (or rejected outright above 8000 px). Screenshots are cut into slices no
  taller than ``chunk_height`` (and no wider than ``max_width``, downscaling
  wider ones) so each is read at full resolution. Slicing needs Pillow; without
  it a screenshot the API would downscale is refused rather than sent. Slices
  overlap by more than a result card is tall, so every card is whole in at
  least one slice.
* **One request per slice** rather than several slices in one request: each
  response stays small (no ``max_tokens`` truncation on a 240-result page), a
  refusal, truncated or malformed answer costs that one slice rather than the
  page (it is logged and kept in ``ClaudeExtractor.failures``), and the model
  never has to reason about which cards two images share.
* **Duplicates are merged in code**, deterministically, and only between
  neighbouring slices: the cards at the bottom of one slice are the cards at
  the top of the next, in the same order, so the two lists are aligned in
  order. A card cut by a slice edge shows only part of its title; it is matched
  by that prefix or suffix plus an equal price or sold date. Two genuinely
  identical sales on one screen stay two listings.
* **Struck-through prices are not sale prices.** When a best offer was accepted
  eBay strikes through the asking price and hides what was paid; like the DOM
  parser, such a card gets ``price=None`` and ``original_price`` = the asking
  price, so it cannot inflate price statistics.
* **Tiles are stitched first.** ``capture.capture_tiles`` overlaps tiles by
  less than a card is tall, so ``extract_tiles`` pastes them back into one
  page before slicing (hot-wheels fixture: 18 requests instead of 23).
* **Challenges stop the run.** If a screenshot shows a CAPTCHA or bot check the
  extractor raises ``LLMExtractionError(reason="captcha")``: a challenge page has
  no prices on it, and the caller should back off rather than keep going. API
  failures that would hit every slice (credentials, rate limit, network) also
  raise at once.

Cost: roughly ``width * height / 750`` input tokens per slice plus ~900 for
the instructions and schema, and ~1-2K output tokens (JSON plus thinking).
``estimate_cost`` (or ``ClaudeExtractor.estimate_cost``) prices screenshots
before you send them. With the default model (claude-opus-5-5, $4 / $20 per
million input / output tokens) a 1280 x 1500 slice is ~3.5K input tokens, about
$0.04, and a 60-90 result page is about $0.65-1.00; DOM parsing is free, which
is why this path is a fallback.

Credentials come from the environment (``ANTHROPIC_API_KEY``, or an
``ant auth login`` profile); nothing here stores or logs a key.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import re
import struct
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import LLMSettings
from .models import Box, Listing
from .normalize import (
    classify_format,
    clean_text,
    parse_count,
    parse_feedback,
    parse_location,
    parse_money,
    parse_shipping,
    parse_sold_date,
)
from .urls import SITE_CURRENCY, normalize_site

log = logging.getLogger(__name__)

ImageInput = bytes | str | Path

# Slice geometry. 1568 px is the long-edge limit of earlier Claude models; at or
# below it no current model downscales a slice, whichever model is configured.
MAX_WIDTH = 1568
MAX_CHUNK_HEIGHT = 1500
# Taller than one result card (s-card layout at 1280 px: median 256 px, max 318
# px on the test fixtures), so every card is whole in at least one of two
# neighbouring slices.
CHUNK_OVERLAP = 400

# Server-side refusal fallback: on a policy decline the API re-runs the request
# on the model Anthropic recommends for that refusal category.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_FALLBACK_MODELS = frozenset({"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"})

# USD per million (input, output) tokens.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
_MAX_IMAGE_TOKENS = 4784  # larger images are downscaled by the API to about this
_PROMPT_TOKENS = 900  # system prompt + instructions + schema, approximately
OUTPUT_TOKENS_PER_REQUEST = 1500  # a guess: ~6 cards x ~150 tokens of JSON, plus low-effort thinking

# Current models read an image at full resolution up to 2576 px on the long edge
# and about 3.75 MP; bigger ones are downscaled, and over 8000 px rejected. Only
# matters without Pillow, when a screenshot cannot be sliced.
_API_MAX_EDGE = 2576
_API_MAX_PIXELS = 3_750_000

# The SDK refuses non-streaming requests above ~21K max_tokens (they may outlast
# its 10-minute timeout). One slice's JSON needs a fraction of this.
_MAX_REQUEST_TOKENS = 16000
_MAX_IMAGE_BYTES = 3_500_000  # the API caps one image at 5 MB of base64

PAGE_ISSUES = ("captcha", "not_search_results", "unreadable")

# Per-card fields the model fills, all verbatim text or null.
TEXT_FIELDS: tuple[str, ...] = (
    "title",
    "price_text",
    "original_price_text",
    "sold_date_text",
    "shipping_text",
    "condition",
    "format_text",
    "bids_text",
    "seller_text",
    "location_text",
)
_FEWER_WORDS = "below_fewer_words_heading"

_FIELD_HINTS: dict[str, str] = {
    "title": "Listing title (all its visible lines), without a 'NEW LISTING' tag",
    "price_text": "Sale price exactly as shown, e.g. '$65.00', '$3.75 to $23.95', 'See price'; "
                  "null when the only price shown is struck through",
    "original_price_text": "A price drawn struck through (with a line through it), e.g. a list price beside "
                           "the sale price, or the asking price of an accepted best offer",
    "sold_date_text": "The sold-date line, e.g. 'Sold  Apr 25, 2026'",
    "shipping_text": "Delivery line, e.g. '+$9.45 delivery', 'Free delivery'",
    "condition": "Condition label under the title, e.g. 'Brand New', 'Pre-Owned', 'Open Box', 'Parts Only'; "
                 "when several grey lines are under the title, the last one",
    "format_text": "Purchase format line, e.g. 'Buy It Now', 'or Best Offer', 'Best offer accepted', 'Auction'",
    "bids_text": "Bid count, e.g. '11 bids'",
    "seller_text": "Seller line with feedback, e.g. 'name 100% positive (267)'",
    "location_text": "Location line, e.g. 'Located in United States'",
}


def _response_schema() -> dict[str, Any]:
    card: dict[str, Any] = {
        "type": "object",
        "properties": {
            **{f: {"type": ["string", "null"], "description": _FIELD_HINTS[f]} for f in TEXT_FIELDS},
            _FEWER_WORDS: {
                "type": "boolean",
                "description": "True if the card is below a 'Results matching fewer words' heading",
            },
        },
        "additionalProperties": False,
    }
    card["required"] = list(card["properties"])
    return {
        "type": "object",
        "properties": {
            "page_issue": {"anyOf": [{"type": "string", "enum": list(PAGE_ISSUES)}, {"type": "null"}]},
            "listings": {"type": "array", "items": card},
        },
        "required": ["page_issue", "listings"],
        "additionalProperties": False,
    }


RESPONSE_SCHEMA = _response_schema()

SYSTEM_PROMPT = (
    "You transcribe screenshots of eBay sold-listing search results into JSON. "
    "You copy the text each result card shows; you never estimate, convert or fill in values."
)

_INSTRUCTIONS = """\
Transcribe every search-result card in this screenshot of {site}, top to bottom.

- Copy text exactly as shown: same characters, currency symbols, spacing and language. \
Do not translate, convert, compute or guess. Use null for anything not shown.
- A price with a line through it is not the sale price: put it in original_price_text. When a best offer \
was accepted, eBay strikes through the asking price and does not show what was paid, so price_text is null.
- A card cut off by the top or bottom edge: include it if at least one line of its title is fully visible, \
copy only the title lines that are fully visible, and use null for every field that is cut off.
- Skip everything that is not a result card: "Shop on eBay" placeholders, ads, filters, related searches, carousels.
- page_issue: "captcha" if the image shows a CAPTCHA, a "verify you are human" or security check, \
"Pardon Our Interruption", "Access Denied" or any other bot-check or block page; "not_search_results" \
if it is some other kind of page (item page, sign-in, error page); "unreadable" if the text cannot be read; \
otherwise null. An image that shows only the header, filters or footer of a results page has \
page_issue null and an empty listings array."""

_SLICE_NOTE = (
    "\n\nThis image is slice {index} of {total} of one tall screenshot. Neighbouring slices overlap, "
    "so a card may also appear in another slice; transcribe it here anyway."
)


class LLMExtractionError(RuntimeError):
    """Claude could not (or must not) extract listings from a screenshot.

    ``reason`` is one of "not_installed", "captcha", "refusal", "truncated",
    "bad_output", "bad_image", "auth", "rate_limited", "api_error",
    "connection", "client_error". ``category`` is the refusal category
    (e.g. "cyber") when ``reason == "refusal"``.
    """

    def __init__(self, message: str, *, reason: str | None = None, category: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.category = category


# Failures caused by what one slice shows (or how the model answered it). Any
# other reason would recur on every slice, so it stops the page instead.
_SLICE_LOCAL_FAILURES = frozenset({"refusal", "truncated", "bad_output"})


# --- images ------------------------------------------------------------------


@dataclass(frozen=True)
class ImageChunk:
    """One slice of a screenshot, ready to send."""

    data: bytes
    media_type: str
    width: int
    height: int
    top: int = 0  # y offset of the slice in the (possibly downscaled) screenshot


@dataclass(frozen=True)
class CostEstimate:
    requests: int
    input_tokens: int
    output_tokens: int
    usd: float | None  # None when the model's price is not in PRICES_PER_MTOK


def _media_type(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


# JPEG start-of-frame markers (they carry the image size); C4, C8 and CC are not frames.
_JPEG_SOF = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})


def _header_size(data: bytes) -> tuple[int, int] | None:
    """``(width, height)`` from the file header of a PNG, JPEG, GIF or WebP, without Pillow."""
    try:
        if data.startswith(b"\x89PNG\r\n\x1a\n") and data[12:16] == b"IHDR":
            return struct.unpack(">II", data[16:24])
        if data.startswith((b"GIF87a", b"GIF89a")):
            return struct.unpack("<HH", data[6:10])
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            kind = data[12:16]
            if kind == b"VP8 ":
                w, h = struct.unpack("<HH", data[26:30])
                return w & 0x3FFF, h & 0x3FFF
            if kind == b"VP8L":
                bits = int.from_bytes(data[21:25], "little")
                return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
            if kind == b"VP8X":
                return int.from_bytes(data[24:27], "little") + 1, int.from_bytes(data[27:30], "little") + 1
            return None
        if data.startswith(b"\xff\xd8"):
            i = 2
            while i + 9 < len(data):
                if data[i] != 0xFF:
                    return None
                marker = data[i + 1]
                if marker == 0xFF:  # fill byte
                    i += 1
                    continue
                if marker in _JPEG_SOF:
                    h, w = struct.unpack(">HH", data[i + 5:i + 9])
                    return w, h
                if marker == 0x01 or 0xD0 <= marker <= 0xD9:  # markers without a length
                    i += 2
                    continue
                i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    except struct.error:
        return None
    return None


def _read_image(image: ImageInput) -> bytes:
    if isinstance(image, (bytes, bytearray, memoryview)):
        return bytes(image)
    path = Path(image)
    try:
        return path.read_bytes()
    except OSError as e:
        raise LLMExtractionError(f"cannot read screenshot {path}: {e}", reason="bad_image") from e


def image_size(image: ImageInput) -> tuple[int, int]:
    """``(width, height)`` of a screenshot, read from its header (Pillow only as a fallback)."""
    data = _read_image(image)
    size = _header_size(data)
    if size:
        return size
    try:
        from PIL import Image
    except ImportError:
        raise LLMExtractionError("cannot read the size of this image (installing Pillow may help: "
                                 "pip install pillow)", reason="bad_image") from None
    try:
        with Image.open(io.BytesIO(data)) as img:
            return img.size
    except Exception as e:  # Pillow raises several unrelated types for bad input
        raise LLMExtractionError(f"cannot read screenshot: {e}", reason="bad_image") from e


def chunk_spans(height: int, chunk_height: int = MAX_CHUNK_HEIGHT,
                overlap: int = CHUNK_OVERLAP) -> list[tuple[int, int]]:
    """``(top, bottom)`` rows of equal, evenly spaced slices covering ``height``.

    Neighbouring slices overlap by at least ``overlap`` rows and none is taller
    than ``chunk_height``. Equal heights avoid a sliver of a last slice that
    would cost a request and show half a card.
    """
    if overlap >= chunk_height:
        raise ValueError("overlap must be smaller than chunk_height")
    if height <= chunk_height:
        return [(0, height)]
    n = math.ceil((height - overlap) / (chunk_height - overlap))
    h = min(chunk_height, math.ceil((height + (n - 1) * overlap) / n))
    tops = [round(i * (height - h) / (n - 1)) for i in range(n)]
    return [(t, t + h) for t in tops]


def _scaled_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    if width <= max_width:
        return width, height
    return max_width, max(1, round(height * max_width / width))


def split_screenshot(
    image: ImageInput,
    *,
    max_width: int = MAX_WIDTH,
    chunk_height: int = MAX_CHUNK_HEIGHT,
    overlap: int = CHUNK_OVERLAP,
) -> list[ImageChunk]:
    """Downscale a too-wide screenshot and cut a too-tall one into overlapping slices.

    An image that already fits is passed through unchanged (no re-encoding).
    Without Pillow nothing can be cut, so an image is sent whole only if the
    API would read it at full resolution; a bigger one raises
    ``LLMExtractionError(reason="not_installed")`` instead of being sent and
    downscaled until its text is unreadable (or rejected, above 8000 px).
    """
    data = _read_image(image)
    media_type = _media_type(data)
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        return [_whole_without_pillow(data, media_type, max_width=max_width, chunk_height=chunk_height)]

    img = _decode(data)
    width, height = img.size
    if (media_type is not None and img.mode in _SENDABLE_MODES and width <= max_width and height <= chunk_height
            and len(data) <= _MAX_IMAGE_BYTES):
        return [ImageChunk(data=data, media_type=media_type, width=width, height=height)]
    return _slice(img, max_width=max_width, chunk_height=chunk_height, overlap=overlap)


def _whole_without_pillow(data: bytes, media_type: str | None, *, max_width: int, chunk_height: int) -> ImageChunk:
    hint = "pip install pillow (it comes with ebay-sold[vision])"
    if media_type is None:
        raise LLMExtractionError(f"unsupported image format; to convert it: {hint}", reason="bad_image")
    size = _header_size(data)
    if size is None:
        raise LLMExtractionError(f"cannot read this image's size without Pillow: {hint}", reason="not_installed")
    w, h = size
    if max(w, h) > _API_MAX_EDGE or w * h > _API_MAX_PIXELS or len(data) > _MAX_IMAGE_BYTES:
        raise LLMExtractionError(
            f"this {w}x{h} px screenshot must be cut into slices before Claude can read it, which needs Pillow: "
            f"{hint}", reason="not_installed")
    if w > max_width or h > chunk_height:
        log.info("Pillow is not installed: sending the %dx%d screenshot whole instead of in slices", w, h)
    return ImageChunk(data=data, media_type=media_type, width=w, height=h)


# Modes sent as they are. Others (CMYK JPEGs, 16-bit PNGs...) are converted, and
# palette / 1-bit images too before resizing, which Pillow would otherwise do with
# nearest-neighbour sampling that garbles digits.
_SENDABLE_MODES = frozenset({"RGB", "RGBA", "L", "LA", "P", "1"})


def _decode(data: bytes) -> Any:
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Image.DecompressionBombError as e:
        raise LLMExtractionError(
            f"screenshot is too large to decode safely ({e}); capture the page in tiles "
            "(ebay_sold.capture.capture_tiles) and use extract_tiles, or raise PIL.Image.MAX_IMAGE_PIXELS",
            reason="bad_image") from e
    except Exception as e:  # Pillow raises several unrelated types for bad input
        raise LLMExtractionError(f"cannot decode screenshot: {e}", reason="bad_image") from e
    return img


def _to_rgb(img: Any) -> Any:
    """RGB or L, with any transparency flattened onto white."""
    from PIL import Image

    if img.mode in ("RGB", "L"):
        return img
    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, "white")
        flat.paste(rgba, mask=rgba.getchannel("A"))
        return flat
    return img.convert("RGB")


def _slice(img: Any, *, max_width: int, chunk_height: int, overlap: int) -> list[ImageChunk]:
    from PIL import Image

    img = _to_rgb(img)
    sw, sh = _scaled_size(img.width, img.height, max_width)
    if (sw, sh) != img.size:
        img = img.resize((sw, sh), Image.Resampling.LANCZOS)
    chunks = []
    for top, bottom in chunk_spans(sh, chunk_height, overlap):
        blob, media_type = _encode(img.crop((0, top, sw, bottom)))
        chunks.append(ImageChunk(data=blob, media_type=media_type, width=sw, height=bottom - top, top=top))
    return chunks


def stitch_tiles(tiles: Sequence[tuple[ImageInput, Box]]) -> Any:
    """Paste ``capture_tiles`` output back into one page image (a Pillow image).

    Tiles overlap by less than a result card is tall, so a card can be cut in
    both; re-slicing the stitched page with ``CHUNK_OVERLAP`` avoids that and
    needs fewer requests than slicing each tile.
    """
    from PIL import Image

    decoded = [(_decode(_read_image(src)), box) for src, box in tiles]
    if not decoded:
        raise ValueError("no tiles")
    first, first_box = decoded[0]
    scale = first.width / first_box.w if first_box.w else 1.0  # screenshot px per CSS px
    origin = min(box.y for _, box in decoded)
    width = max(img.width for img, _ in decoded)
    height = max(round((box.y - origin) * scale) + img.height for img, box in decoded)
    page = Image.new("RGB", (width, height), "white")
    for img, box in decoded:
        page.paste(img.convert("RGB"), (0, round((box.y - origin) * scale)))
    return page


def _encode(img: Any) -> tuple[bytes, str]:
    # PNG keeps text edges sharp; fall back to JPEG only for photo-heavy slices
    # that would exceed the API's per-image size limit.
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    if buf.tell() <= _MAX_IMAGE_BYTES:
        return buf.getvalue(), "image/png"
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "image/jpeg"


def image_tokens(width: int, height: int) -> int:
    """Approximate input tokens Claude bills for one image."""
    return min(math.ceil(width * height / 750), _MAX_IMAGE_TOKENS)


def estimate_cost(
    image_sizes: Iterable[tuple[int, int]],
    *,
    model: str = "claude-opus-5-5",
    max_width: int = MAX_WIDTH,
    chunk_height: int = MAX_CHUNK_HEIGHT,
    overlap: int = CHUNK_OVERLAP,
    output_tokens_per_request: int = OUTPUT_TOKENS_PER_REQUEST,
) -> CostEstimate:
    """Rough cost of extracting screenshots of the given ``(width, height)`` sizes.

    Uses the same slicing as ``split_screenshot``. Output tokens are a guess
    (they include the model's thinking), so treat the result as an estimate.
    """
    requests = input_tokens = 0
    for width, height in image_sizes:
        sw, sh = _scaled_size(width, height, max_width)
        for top, bottom in chunk_spans(sh, chunk_height, overlap):
            requests += 1
            input_tokens += image_tokens(sw, bottom - top) + _PROMPT_TOKENS
    output_tokens = requests * output_tokens_per_request
    price = PRICES_PER_MTOK.get(model)
    usd = None
    if price:
        usd = round((input_tokens * price[0] + output_tokens * price[1]) / 1_000_000, 4)
    return CostEstimate(requests=requests, input_tokens=input_tokens, output_tokens=output_tokens, usd=usd)


# --- merging and normalization -------------------------------------------------

_NEW_LISTING_RE = re.compile(r"^\s*new listing\b\s*", re.IGNORECASE)
_SR_SUFFIX_RE = re.compile(r"\s*opens in a new window or tab\s*$", re.IGNORECASE)
_NON_WORD_RE = re.compile(r"[\W_]+")
# eBay strikes through the asking price of an accepted best offer and hides what was paid.
_OFFER_ACCEPTED_RE = re.compile(
    r"offer accepted|preisvorschlag angenommen|offre (?:directe )?accept|offerta accettata|oferta aceptada",
    re.IGNORECASE,
)


def _clean_title(title: str | None) -> str:
    return _SR_SUFFIX_RE.sub("", _NEW_LISTING_RE.sub("", clean_text(title)))


def _norm(text: str | None) -> str:
    return _NON_WORD_RE.sub(" ", clean_text(text).casefold()).strip()


def _norm_title(entry: dict[str, Any]) -> str:
    return _norm(_clean_title(entry["title"]))


def _conflict(a: dict[str, Any], b: dict[str, Any], field: str) -> bool:
    return a[field] is not None and b[field] is not None and _norm(a[field]) != _norm(b[field])


def _agree(a: dict[str, Any], b: dict[str, Any], field: str) -> bool:
    return a[field] is not None and b[field] is not None and _norm(a[field]) == _norm(b[field])


_EXACT, _PARTIAL = 2, 1


def _match_kind(a: dict[str, Any], b: dict[str, Any]) -> int:
    """How two sightings in neighbouring slices can be the same card: 0 (not), _PARTIAL or _EXACT.

    _EXACT: same title, and price / date agree wherever both are visible.
    _PARTIAL: one title is the other's first or last lines, as when a slice
    edge cuts through it; then nothing visible in both may differ, and the
    price (cut at the top) or the sold date (cut at the bottom) must match.
    """
    ta, tb = _norm_title(a), _norm_title(b)
    if not ta or not tb or _conflict(a, b, "price_text") or _conflict(a, b, "sold_date_text"):
        return 0
    if ta == tb:
        return _EXACT
    short, long = sorted((ta, tb), key=len)
    if not (long.startswith(short + " ") or long.endswith(" " + short)):
        return 0
    if _conflict(a, b, "shipping_text") or _conflict(a, b, "seller_text"):
        return 0
    return _PARTIAL if _agree(a, b, "price_text") or _agree(a, b, "sold_date_text") else 0


def _filled(entry: dict[str, Any]) -> int:
    return sum(entry[f] is not None for f in TEXT_FIELDS)


def _combine(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge two sightings of one card, preferring the more complete one."""
    best, other = (a, b) if _filled(a) >= _filled(b) else (b, a)
    merged = {f: best[f] if best[f] is not None else other[f] for f in TEXT_FIELDS}
    if _norm_title(a) != _norm_title(b):  # a title cut by a slice edge: keep the whole one
        merged["title"] = max(a["title"], b["title"], key=lambda t: len(_norm(_clean_title(t))))
    merged[_FEWER_WORDS] = bool(a[_FEWER_WORDS] or b[_FEWER_WORDS])
    return merged


def _align(prev: Sequence[dict[str, Any]], cur: Sequence[dict[str, Any]]) -> dict[int, int]:
    """Pair cards of a slice with cards of the previous slice: ``{cur index: prev index}``.

    The overlap shows the same cards in the same order, so this is an
    order-preserving alignment (weighted LCS) that maximises the number of
    pairs, then the number of exact pairs. Ties go to pairs near the bottom of
    ``prev`` and the top of ``cur``, which is where the overlap is: of two
    identical sales, the one at the slice edge is the one seen again.
    """
    n, m = len(prev), len(cur)
    if not n or not m:
        return {}
    kinds = [[_match_kind(p, c) for c in cur] for p in prev]

    def gain(i: int, j: int, kind: int) -> tuple[int, int, int]:  # 1-based i, j
        return 1, int(kind == _EXACT), i + (m - j + 1)

    best = [[(0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = max(best[i - 1][j], best[i][j - 1])
            if kinds[i - 1][j - 1]:
                g, b = gain(i, j, kinds[i - 1][j - 1]), best[i - 1][j - 1]
                score = max(score, (b[0] + g[0], b[1] + g[1], b[2] + g[2]))
            best[i][j] = score
    pairs: dict[int, int] = {}
    i, j = n, m
    while i and j:
        kind = kinds[i - 1][j - 1]
        if kind:
            g, b = gain(i, j, kind), best[i - 1][j - 1]
            if best[i][j] == (b[0] + g[0], b[1] + g[1], b[2] + g[2]):
                pairs[j - 1] = i - 1
                i, j = i - 1, j - 1
                continue
        if best[i][j] == best[i - 1][j]:
            i -= 1
        else:
            j -= 1
    return pairs


def merge_chunks(per_chunk: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Merge per-slice card lists, dropping the duplicates slice overlaps create.

    A card may match at most one card of the *previous* slice; cards within one
    slice are never merged with each other. An empty slice (nothing on it, or a
    slice that could not be read) breaks the chain.
    """
    out: list[dict[str, Any]] = []
    prev: list[int] = []
    for entries in per_chunk:
        pairs = _align([out[i] for i in prev], entries)
        current: list[int] = []
        for j, entry in enumerate(entries):
            if j in pairs:
                target = prev[pairs[j]]
                out[target] = _combine(out[target], entry)
                current.append(target)
            else:
                out.append(entry)
                current.append(len(out) - 1)
        prev = current
    return out


def to_listing(raw: dict[str, Any], *, site: str = "www.ebay.com", today: date | None = None,
               position: int | None = None, matches_query: bool = True) -> Listing | None:
    """Turn one transcribed card into a ``Listing`` using ``ebay_sold.normalize`` only."""
    site = normalize_site(site)
    title = _clean_title(raw.get("title"))
    if not title:
        return None
    text = {f: clean_text(raw.get(f)) or None for f in TEXT_FIELDS if f != "title"}
    currency = SITE_CURRENCY[site]
    price_text, original_text = text["price_text"], text["original_price_text"]
    # The only price shown is struck through (an accepted best offer): it is the
    # asking price, not what was paid. Like the DOM parser, price stays None and
    # price_text keeps what the card shows. The model may report it as the
    # struck price only (as asked), as both, or (missing the line) as the price.
    struck_only = bool(
        (price_text is None and original_text)
        or (price_text and original_text and _norm(price_text) == _norm(original_text))
        or (price_text and not original_text and _OFFER_ACCEPTED_RE.search(text["format_text"] or ""))
    )
    if struck_only:
        price_text = price_text or original_text
        money, original = None, parse_money(price_text, currency)
    else:
        money, original = parse_money(price_text, currency), parse_money(original_text, currency)
    if original is not None and money is not None and original.amount == money.amount:
        original = None
    # "Pre-Owned · Nintendo": the condition comes first (as in the DOM parser).
    condition = clean_text((text["condition"] or "").split("·")[0]) or None
    listing_format, bids = classify_format([t for t in (text["format_text"], text["bids_text"]) if t])
    if bids is None and text["bids_text"]:
        bids = parse_count(text["bids_text"])
    seller, feedback_pct, feedback_count = (
        parse_feedback(text["seller_text"]) if text["seller_text"] else (None, None, None))
    return Listing(
        item_id=None,
        site=site,
        title=title,
        price=money.amount if money else None,
        price_max=money.amount_max if money else None,
        original_price=original.amount if original else None,
        currency=(money or original).currency if (money or original) else currency,
        price_text=price_text,
        shipping=parse_shipping(text["shipping_text"], currency),
        shipping_text=text["shipping_text"],
        sold_date=parse_sold_date(text["sold_date_text"], today, day_first=site != "www.ebay.com"),
        sold_date_text=text["sold_date_text"],
        condition=condition,
        listing_format=listing_format,  # type: ignore[arg-type]
        bids=bids,
        seller=seller,
        seller_feedback_pct=feedback_pct,
        seller_feedback_count=feedback_count,
        location=parse_location(text["location_text"]),
        matches_query=matches_query,
        position=position,
        extraction="llm",
        confidence=None,
    )


def _coerce_entry(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    entry: dict[str, Any] = {}
    for f in TEXT_FIELDS:
        value = item.get(f)
        entry[f] = (clean_text(value if isinstance(value, str) else str(value)) or None) if value is not None else None
    entry[_FEWER_WORDS] = item.get(_FEWER_WORDS) is True
    return entry if entry["title"] else None


# --- the extractor -------------------------------------------------------------


class ClaudeExtractor:
    """Extract sold listings from screenshots with Claude.

    ``client`` may be any object with ``beta.messages.create(**kwargs)``
    (tests pass a fake). By default an ``anthropic.Anthropic()`` client is
    created, which reads credentials from the environment.

    ``server_fallback`` sends ``fallbacks="default"`` so a request the model
    declines is retried server-side on Anthropic's recommended fallback model.
    It defaults to on for models that support it on the Claude API; turn it
    off for clients pointed at Bedrock, Vertex AI or Foundry.

    ``usage`` accumulates requests and tokens across calls. Use
    ``estimate_cost`` to price screenshots before sending them.

    A slice Claude declines, or answers with truncated or malformed JSON, is
    skipped with a warning and its error kept in ``failures`` (reset by every
    extract call), so one bad slice does not discard the rest of the page. If
    every slice fails, the first error is raised. A CAPTCHA, and API errors
    that would affect every slice (auth, rate limit, network), raise at once.
    """

    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        client: Any = None,
        server_fallback: bool | None = None,
        max_width: int = MAX_WIDTH,
        chunk_height: int = MAX_CHUNK_HEIGHT,
        overlap: int = CHUNK_OVERLAP,
    ) -> None:
        if overlap >= chunk_height:
            raise ValueError("overlap must be smaller than chunk_height")
        self.settings = settings or LLMSettings()
        if client is None:
            try:
                import anthropic
            except ImportError as e:
                raise LLMExtractionError(
                    "Claude extraction needs the Anthropic SDK: pip install 'ebay-sold[llm]'",
                    reason="not_installed",
                ) from e
            try:
                client = anthropic.Anthropic()
            except anthropic.AnthropicError as e:  # e.g. a configured profile that does not exist
                raise LLMExtractionError(f"cannot set up Claude API credentials: {e}", reason="auth") from e
            # Without any credential the SDK fails later with a bare TypeError; say what to do now.
            if not any(getattr(client, attr, None) for attr in ("api_key", "auth_token", "credentials")):
                raise LLMExtractionError(
                    "no Claude API credentials found: set ANTHROPIC_API_KEY or run `ant auth login`",
                    reason="auth",
                )
        self.client = client
        self.server_fallback = self.settings.model in _FALLBACK_MODELS if server_fallback is None else server_fallback
        self.max_width = max_width
        self.chunk_height = chunk_height
        self.overlap = overlap
        self.usage: dict[str, int] = {"requests": 0, "input_tokens": 0, "output_tokens": 0}
        self.failures: list[LLMExtractionError] = []

    # -- public API --

    def extract(self, image: ImageInput, *, site: str = "www.ebay.com", today: date | None = None) -> list[Listing]:
        """Listings visible in one screenshot (any height), top to bottom.

        Raises ``LLMExtractionError`` if the screenshot shows a CAPTCHA or bot
        check, if Claude declines, or if the API call fails.
        """
        return self.extract_many([image], site=site, today=today)

    def extract_many(self, images: Iterable[ImageInput], *, site: str = "www.ebay.com",
                     today: date | None = None) -> list[Listing]:
        """Listings from several screenshots of one page, in reading order.

        Consecutive images are treated like slices of one page: a card seen at
        the bottom of one and the top of the next is returned once.
        """
        chunks = [c for image in images for c in self._split(image)]
        return self._extract_chunks(chunks, site=site, today=today)

    def extract_tiles(self, tiles: Sequence[tuple[ImageInput, Box]], *, site: str = "www.ebay.com",
                      today: date | None = None) -> list[Listing]:
        """Listings from ``capture.capture_tiles`` output (``(png_path, tile_box)`` pairs).

        The tiles are stitched back into one page and re-sliced, so no card is
        cut at a tile boundary. Without Pillow this falls back to ``extract_many``.
        """
        if not tiles:
            return []
        try:
            import PIL  # noqa: F401
        except ImportError:
            return self.extract_many([src for src, _ in tiles], site=site, today=today)
        page = stitch_tiles(tiles)
        chunks = _slice(page, max_width=self.max_width, chunk_height=self.chunk_height, overlap=self.overlap)
        return self._extract_chunks(chunks, site=site, today=today)

    def estimate_cost(self, images: Iterable[ImageInput], *,
                      output_tokens_per_request: int = OUTPUT_TOKENS_PER_REQUEST) -> CostEstimate:
        """Rough cost of ``extract_many(images)`` with this extractor's model and slicing."""
        return estimate_cost(
            [image_size(i) for i in images],
            model=self.settings.model,
            max_width=self.max_width,
            chunk_height=self.chunk_height,
            overlap=self.overlap,
            output_tokens_per_request=output_tokens_per_request,
        )

    # -- internals --

    def _extract_chunks(self, chunks: list[ImageChunk], *, site: str, today: date | None) -> list[Listing]:
        site = normalize_site(site)
        self.failures = []
        per_chunk: list[list[dict[str, Any]]] = []
        for i, chunk in enumerate(chunks, 1):
            try:
                per_chunk.append(self._read_chunk(chunk, index=i, total=len(chunks), site=site))
            except LLMExtractionError as e:
                if e.reason not in _SLICE_LOCAL_FAILURES:
                    raise
                log.warning("skipping a slice Claude could not read: %s", e)
                self.failures.append(e)
                per_chunk.append([])
        if self.failures and len(self.failures) == len(chunks):
            raise self.failures[0]
        if self.failures:
            log.warning("%d of %d slice(s) could not be read; their listings are missing", len(self.failures),
                        len(chunks))
        listings: list[Listing] = []
        below_fewer_words = False
        for raw in merge_chunks(per_chunk):
            # Everything after eBay's "Results matching fewer words" heading is a loose match.
            below_fewer_words = below_fewer_words or raw[_FEWER_WORDS]
            listing = to_listing(raw, site=site, today=today, position=len(listings) + 1,
                                 matches_query=not below_fewer_words)
            if listing is not None:
                listings.append(listing)
        log.info("Claude read %d listings from %d slice(s)", len(listings), len(chunks))
        return listings

    def _split(self, image: ImageInput) -> list[ImageChunk]:
        return split_screenshot(image, max_width=self.max_width, chunk_height=self.chunk_height, overlap=self.overlap)

    def _request_kwargs(self, chunk: ImageChunk, *, index: int, total: int, site: str,
                        max_tokens: int) -> dict[str, Any]:
        prompt = _INSTRUCTIONS.format(site=site)
        if total > 1:
            prompt += _SLICE_NOTE.format(index=index, total=total)
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}}
        if self.settings.effort:
            output_config["effort"] = self.settings.effort
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": chunk.media_type,
                            "data": base64.standard_b64encode(chunk.data).decode("ascii"),
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            "output_config": output_config,
        }
        if self.server_fallback:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def _read_chunk(self, chunk: ImageChunk, *, index: int, total: int, site: str) -> list[dict[str, Any]]:
        max_tokens = min(self.settings.max_tokens, _MAX_REQUEST_TOKENS)
        if max_tokens < self.settings.max_tokens and index == 1:
            log.warning("llm.max_tokens=%d is above what a non-streaming request allows; using %d per slice",
                        self.settings.max_tokens, max_tokens)
        response = self._create(self._request_kwargs(chunk, index=index, total=total, site=site, max_tokens=max_tokens))
        if response.stop_reason == "max_tokens":
            retry_tokens = min(max_tokens * 2, _MAX_REQUEST_TOKENS)
            if retry_tokens > max_tokens:
                log.info("slice %d/%d hit max_tokens=%d; retrying with %d", index, total, max_tokens, retry_tokens)
                response = self._create(
                    self._request_kwargs(chunk, index=index, total=total, site=site, max_tokens=retry_tokens))
        return self._parse_response(response, index=index, total=total)

    def _parse_response(self, response: Any, *, index: int, total: int) -> list[dict[str, Any]]:
        where = f"slice {index}/{total}" if total > 1 else "screenshot"
        served_by = getattr(response, "model", None)
        if served_by and served_by != self.settings.model:
            log.info("%s was served by %s (requested %s; a refusal fallback or alias)", where, served_by,
                     self.settings.model)
        # Check stop_reason before content: a refusal's content is empty or partial.
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            explanation = getattr(details, "explanation", None)
            raise LLMExtractionError(
                f"Claude declined to read the {where} (category: {category or 'unspecified'})"
                + (f": {explanation}" if explanation else ""),
                reason="refusal",
                category=category,
            )
        if response.stop_reason == "max_tokens":
            raise LLMExtractionError(
                f"Claude's answer for the {where} was cut off at max_tokens; raise llm.max_tokens or use a "
                "smaller chunk_height",
                reason="truncated",
            )
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise LLMExtractionError(f"Claude returned no text for the {where}", reason="bad_output")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMExtractionError(f"Claude returned invalid JSON for the {where}: {e}", reason="bad_output") from e
        if not isinstance(payload, dict) or not isinstance(payload.get("listings", []), list):
            raise LLMExtractionError(f"unexpected JSON shape for the {where}", reason="bad_output")

        issue = payload.get("page_issue")
        if issue == "captcha":
            raise LLMExtractionError(
                f"the {where} shows a CAPTCHA / bot check, not results; stop and retry later",
                reason="captcha",
            )
        if issue in ("not_search_results", "unreadable"):
            log.warning("Claude reports the %s is %s", where, issue.replace("_", " "))
            return []
        entries = [e for e in map(_coerce_entry, payload.get("listings") or []) if e is not None]
        log.debug("%s: %d cards", where, len(entries))
        return entries

    def _create(self, kwargs: dict[str, Any]) -> Any:
        try:
            import anthropic
        except ImportError:  # an injected client without the SDK installed
            anthropic = None  # type: ignore[assignment]
        if anthropic is None:
            response = self.client.beta.messages.create(**kwargs)
        else:
            try:
                response = self.client.beta.messages.create(**kwargs)
            # Most specific first. The SDK has already retried 408/409/429/5xx
            # and connection errors (max_retries) before any of these reach us.
            except anthropic.AuthenticationError as e:
                raise LLMExtractionError(
                    "the Claude API rejected the credentials; set ANTHROPIC_API_KEY or run `ant auth login`",
                    reason="auth") from e
            except anthropic.RateLimitError as e:
                retry_after = e.response.headers.get("retry-after")
                raise LLMExtractionError(
                    "rate limited by the Claude API" + (f"; retry after {retry_after}s" if retry_after else ""),
                    reason="rate_limited") from e
            except anthropic.APIStatusError as e:
                raise LLMExtractionError(
                    f"Claude API error {e.status_code}: {e.message} (request id {e.request_id})",
                    reason="api_error") from e
            except anthropic.APIConnectionError as e:
                raise LLMExtractionError(f"cannot reach the Claude API: {e}", reason="connection") from e
            except anthropic.AnthropicError as e:  # e.g. no credentials configured at all
                raise LLMExtractionError(
                    f"Claude client error: {e} (set ANTHROPIC_API_KEY or run `ant auth login`)",
                    reason="client_error") from e
            except ValueError as e:  # raised by the SDK before sending, e.g. "Streaming is required..."
                raise LLMExtractionError(f"Claude client error: {e}", reason="client_error") from e
        usage = getattr(response, "usage", None)
        self.usage["requests"] += 1
        self.usage["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        self.usage["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        return response
